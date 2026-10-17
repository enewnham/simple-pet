# Session liveness and repository awareness

Give the pet reliable evidence of agent activity and pending work. Keep process presence,
activity freshness, local changes, and PR delivery separate. Pending repository work can
inform commentary while the pet rests. Retain historical memory as supporting context.

## Implementation sequence

- [x] P0 Establish durable progress tracking.
- [x] P1 Extract attention and context, with application-owned memory.
- [x] P2 Fair session scheduling and session-scoped waiting suppression.
- [x] P3 Verified process ownership and asynchronous liveness probes.
- [x] P4 Separate presence from activity freshness after five quiet minutes.
- [x] P5 Asynchronous Git and GitHub cached repository tracking.
- [x] P6 Persist associations through additive migration and derive immutable work views.
- [x] P7 Prioritize input requests, failures, recent activity, pending work, completion, history.
- [x] P8 Validate commentary against changing subjects and update diagnostics and guidance.
- [x] P9 Run scoped checks, make check, and review integration.

## Architecture

```text
pet_daemon/
├── daemon.py                # Lifecycle, sockets, and coordination
├── sessions.py              # Sessions, turns, calls, and owner associations
├── processes.py             # Process identity discovery and liveness
├── repositories/
│   ├── __init__.py          # Tracker exports
│   ├── tracker.py           # Associations, cached observations, refresh scheduling
│   ├── git.py               # Local Git commands and parsing
│   └── github.py            # PR discovery and queries
├── work.py                  # Derived immutable session and repository views
├── attention.py             # Subject selection, fairness, and cooldowns
├── context.py               # Knowledge selection and bounded model payloads
├── cognition.py             # Model requests and response validation
└── memory.py                # Persistent facts, summaries, and associations
```

Animation, policy, execution, and episode modules retain their responsibilities.

## Step requirements

P0 records the checkout, baseline, validation, findings, and next action. Update progress
after each coherent result and before handoff. Resume by reading this file and the diff.

P1 extracts scheduling, cooldowns, and subject selection into attention. Context owns
payloads, instructions, history selection, and budgeting. The application owns memory
independently of cognition. Preserve behavior during extraction.

P2 rotates equally eligible sessions fairly, marks only the selected session checked,
scopes waiting suppression to its session, and uses recent completed work for completion
context.

P3 discovers persistent harness or wrapper owners through verified ancestry, captures PID
and start time, deduplicates probes, and checks about every ten seconds. Sessions retire
on confirmed owner exit or identity replacement. Shared hosts and subagents preserve hook
lifecycle distinctions.

P4 keeps owner liveness independent of working state. Quiet turns without outstanding calls
become uncertain after five minutes and release idle care and naps. Preserve stale-turn
cleanup and timeout handling for ownerless sessions.

P5 discovers worktrees and associates sessions. Observe branch, HEAD, staged, unstaged,
untracked changes, conflicts, upstream, and local ahead/behind counts. Observe associated
PR state, merge information, reviews, and checks. Refresh Git every 30 seconds and GitHub
every three minutes while relevant.

P6 preserves many-to-many session/worktree and PR associations across exits, branch
switches, and restart. Derive immutable work views from snapshots. Migrate storage without
destroying existing history, speech, summaries, or contexts.

P7 ranks explicit input requests and fresh failures first, then recent activity, recent
pending repository work, newly observed completion, and relevant history. Unchanged polling
does not increase recency. A merged PR resolves its pending state while local changes
remain separate.

P8 rejects commentary when its subject changes during generation, including a PR merge.
Log retirement, observation changes, refresh failures, and attention selection. Update
configuration examples, README usage guidance, and the codebase guide.

P9 runs scoped checks and make check, and reviews ownership, asynchronous result validity,
persistence, and idle behavior against every acceptance case.

## Contracts

Hooks accept optional owner PID and start time. Probes report alive, exited, replaced,
or unknown. Sessions decide retirement. Shared hosts and subagents retain hook lifecycle
semantics. Quiet turns without calls cease blocking rest after five minutes. Ownerless
sessions retain stale cleanup.

Repository observations include identity, timestamps, revision, and refresh status.
Local Git refreshes every 30 seconds, GitHub every three minutes. External commands have
timeouts and bounded concurrency. Animation reads caches. Stale branch or generation
results cannot overwrite current observations. Failed probes retain the last successful
observation and its age. Unchanged polling does not advance evidence recency.

Work views derive from authoritative snapshots. Attention chooses subjects and reasons.
Context owns bounded payloads and history selection. Cognition owns model requests and
validation. Application owns memory. Preserve the 16,000-byte default request budget.
Model responses never establish process or PR state. Merged PRs resolve their own pending
state while remaining local changes stay separate. Reject commentary if its subject
changes during generation.

## Verification

Use temporary Git repositories, short-lived processes, and a simulated GitHub command
boundary. Cover missing exit hooks, PID reuse, intermediary shells, shared hosts, unknown
ownership, quiet live agents, outstanding tools, pending work after exit, manual commits
and merges, merged PRs with edits, branch switches and concurrent refreshes, fair attention,
unchanged polling, query failures, restart persistence, migration preservation, and work
resolved during generation.

## Progress and compaction handoff

Current step: complete.
Completed steps: P0–P9.
Checkout and branch: /Users/elliot/sc/simple-pet, main, baseline 77054ff.
Baseline changes: clean. User committed logging changes and cleared uncommitted files.
Files changed: attention/context/process/repository/work modules, daemon, sessions, behavior, memory, config, hook clients, tests, and this plan.
Decisions added or revised: preserve explicit input waits and outstanding tools as engaged. Quiet turns without calls become uncertain. Use additive v4→v5 migration. Keep cognition helper compatibility while the application owns its Memory instance. Poll local Git using one porcelain snapshot plus identity verification. Keep last successful observation timestamps separate from retry timestamps.
Validation run and results: make check passes. Ran 295 Python tests with one desktop-only check skipped, Swift core checks, Swift asset-import checks, and shell syntax checks. Scoped process, repository, migration, context-budget, and runtime tests pass. git diff --check passes. Initial check-in and prompt assertion failures were corrected to match uncertain activity and the parallel agent's speech-only prompt changes. Process tests use unsandboxed ps access on macOS.
Outstanding findings: independent reviews reconciled. Fixed terminal PR rediscovery, fresh CI failure priority, stable recency, explicit/care direction preservation, retry after failed generation, and uncertainty invalidation. Recent pending repository horizon is 24 hours, fresh failure/completion horizon is five minutes. No remaining review findings.
Next concrete action: none. Implementation, verification, and review are complete.
