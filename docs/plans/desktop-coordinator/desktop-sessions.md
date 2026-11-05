# Desktop sessions and adapters

- Status: Implementation in progress
- Date: 2026-11-05
- Implementation: Public adapter verified, personal desktop acceptance pending

## Context

The pet observes coding tools through hooks and can open recorded worktrees, PRs,
and sessions through [navigation handlers](../../../pet_daemon/navigation.py).
Opening a destination does not establish that a new harness session exists or that
the pet owns it. The [desktop adapter](../../../pet_daemon/desktop/) records explicit
delegation for starting and maintaining work on the user's desktop.

This specification owns that boundary. [The overview](overview.md) contains the
delivery plan. [Responsibilities and evidence](responsibilities-and-evidence.md)
defines authority and completion evidence. [SDK workflows](sdk-workflows.md) defines
the internal coordinator. [Conversation](conversation.md) defines the human controls.

## Decision

Use a configured external desktop adapter to prepare a workspace and manage a
native agent harness session. The adapter chooses its terminal, worktree tools,
bootstrap steps, harness executable, and provider-specific lifecycle operations.

The public implementation owns task intent, operation identity, accepted results,
and coordination policy. It invokes the adapter with validated data. Claude Code,
Codex, Cursor, and other desktop harnesses can participate without changing the
pet's internal SDK vendor.

The internal pet performs coordination and research. Code implementation happens
in the external harness, under that harness's instructions, skills, configuration,
and normal permission handling.

The pet's internal SDK model, thinking mode, and effort settings do not determine
the provider, model, or effort used by a desktop harness. Those choices remain in
the configured desktop adapter and native harness configuration.

## Configuration and capabilities

Configure an executable argument vector for the adapter. Pass a versioned JSON
request on standard input and read a JSON result from standard output. Keep adapter
diagnostics on standard error. Invoke the configured executable directly, without
interpolating model-provided text into shell commands.

Configuration names supported operations and operation timeouts. It also identifies
any repository/workspace mappings needed by the adapter. Keep credentials and
personal commands in user configuration. Public examples must not contain personal
home paths, organization-specific setup, or assumptions about a terminal.

Automatic launch requires launch and inspect. Automatic follow-through additionally
requires continue. Open and stop are optional capabilities. Hide unsupported actions
or explain their unavailability in the conversation. A harness that cannot accept
automatic continuation still supports observation, navigation, and a prepared human
handoff.

Timeouts bound how long the pet waits for an adapter operation. They do not establish
whether an external operation happened. The adapter may start asynchronous workspace
preparation and expose its progress through inspect.

## Task and operation records

A task records its responsibility, bounded objective, expected result, repository
context, evidence references, constraints, and the revision of the accepted scope.
The user can inspect its brief. The application establishes that the responsibility
permits delegation before requesting launch.

An operation has its own stable command ID, task ID, operation kind, relevant task
and evidence revisions, attempt state, and timestamps. Continuing a session creates
a new operation. Retrying or inspecting the same uncertain operation retains its
command ID.

The application persists operation intent before invoking the adapter. A receipt
records the adapter's external identity, resulting workspace, harness identity, and
native session locator when available. Adapter and session identities are opaque
values at this boundary. An SDK conversation ID is not a desktop harness identity.

Keep operation outcome separate from task execution and goal completion. Useful
operation outcomes include pending, accepted, failed, and unknown. Useful task
observations include preparing, running, ready for another instruction, waiting for
human input, stopped, and execution ended. An unsupported or unknown readiness
observation cannot authorize automatic continuation.

## Protocol operations

Package 1 defines serialized request/result fixtures for these semantic contracts.
Version the protocol explicitly. Preserve the distinction between absent, unknown,
failed, and completed information in those fixtures.

| Operation | Input meaning | Result meaning |
| --- | --- | --- |
| Launch | Protocol version, command/task identity, accepted scope revision, bounded brief, repository context, supporting references, constraints, and expected result. | Acceptance or preparation state, external launch identity, workspace, and harness/session locator when established. |
| Inspect | Task identity and any command identity whose outcome needs reconciliation. | Known operation outcome, external/session identity, current task state, readiness, and observation time. Unknown remains explicit. |
| Continue | A new command identity, owned task/session identity, bounded instruction, accepted scope revision, and expected readiness/state. | Whether the instruction was accepted, is still pending, failed, or cannot be delivered because the state is busy/stale/unknown. |
| Open | An established task/session identity and destination. | Whether the requested navigation was dispatched or failed. This does not establish task progress. |
| Stop | An established owned task/session identity and a command identity. | Stop request acceptance and, when observed, stopped execution. It does not undo completed work. |

Adapters retain enough task/command mapping to inspect prior dispatches after a
daemon restart. Deduplicate repeated requests with the same command identity. If
the underlying harness cannot establish whether an instruction was delivered, report
unknown and preserve the mapping. The pet does not invent exactly-once guarantees
from terminal writes or process exit codes.

## Launch, observation, and ownership

The launch flow is:

1. Validate the adopted responsibility, selected repository, task brief, and current
   scope. Persist the task and launch operation.
2. Ask the configured adapter to prepare the workspace and launch the session.
3. Record its receipt or preserve an uncertain outcome for inspection.
4. Associate native hooks and adapter task events with the explicit delegation ID.
5. Expose the resulting workspace, session, and state in conversation and Activity.

Extend the existing [hook adapter](../../../scripts/pet-hook) and accepted hook
contract to carry a delegation ID through a documented environment value or task
event interface. The launcher supplies the value to its harness process. Validate
events against recorded task/operation identity before accepting ownership or effects.
An unrecognized identifier cannot create an adopted responsibility.

A hook can arrive before a launch receipt, or a receipt before the first hook.
Reconcile both orderings into the same task. Keep provider-native session aliases
and parent/subagent observations as evidence without broadening control to unrelated
sessions. Sharing a checkout, branch, PR, ticket, or process ancestry does not confer
pet ownership.

The pet can understand configured, visible work, including manually started sessions.
It can automatically continue a harness session only when it owns that session and
has applicable authority. In this delivery, ownership comes from launch through the
configured adapter and its recorded responsibility. The user can inspect and direct
that work through the native session.

## Continuing owned work

After a task reports a result, compare the evidence with its expected outcome. A
turn ending or a success statement does not establish that the responsibility is
complete. Record the result and determine whether another bounded step is within
the accepted scope.

An owned-session result or readiness change creates or updates a candidate in the
[coordination loop](responsibilities-and-evidence.md#ongoing-coordination-loop).
The event does not dispatch a continuation directly. The application coalesces
related changes and applies timing, capacity, priority, and fairness before reasoning
about the next step. Fair service for queued work does not stop a running harness.

Before continue, recheck responsibility revision, pause state, ownership, evidence,
and session readiness. The adapter checks that the intended harness can accept the
instruction and returns busy, stale, or unknown when it cannot establish delivery.
Do not paste an instruction into an arbitrary foreground shell or a session merely
matched by its directory.

The user's submitted directions in a harness supersede pending pet instructions
that conflict with them. Reassess any changed task scope before another automatic
step. Questions and priority changes outside the adopted responsibility return to
the user through the conversation. Routine next steps within that responsibility
do not require a new confirmation for every turn.

## Pause, stop, sleep, and recovery

Pausing coordination prevents new automatic operations for the selected
responsibility and invalidates queued ones. It does not claim to stop a running
harness. Stop is a separate supported adapter operation. Closing the conversation
does neither.

On system sleep, suspend coordination and cancel appropriate internal work through
the application's lifecycle. External harnesses may continue according to the OS
and their own processes. On wake or restart, inspect unresolved operations and
owned sessions before considering another instruction. Reevaluate time-based
follow-ups against current evidence.

If launch succeeds before a receipt is committed, inspect the same command ID.
If a continuation was accepted before its response was lost, inspect that command
before considering another submission. A timeout or disconnected socket is an
uncertain outcome. Automatic retry cannot turn it into a new operation.

After responsibility revisions or pauses, retain uncertain-operation records for
reconciliation. A late receipt can establish that work exists without authorizing
further follow-through. Show that distinction to the user.

The repository's schema reset policy clears local coordination state and authority.
Do not reconstruct authority merely because a surviving external session or adapter
record can be found. Restart recovery assumes the corresponding local records
remain within the same schema version.

## Personal adapter integration

The user's personal adapter composes the existing worktree helpers, Kitty, a
persistent terminal wrapper, and the chosen harness. It creates a fresh worktree
and a fresh terminal session for the delegated task. It retains task/command
mapping, returns the workspace and native session identity, and implements the
capabilities it can establish reliably.

Existing focus helpers can revisit a matching session and may paste text without
submitting it. The personal implementation must distinguish navigation, prepared
handoff, and actual launch/continuation. Preserve its normal worktree bootstrap and
harness instructions. Keep this adapter outside the public repository.

The public delivery includes protocol documentation, example configuration,
fake-adapter fixtures, and an integration guide. It does not require Kitty or the
user's shell functions for its automated checks.

## Invariants

- Record operation intent before requesting an external effect.
- Correlate effects through task and command identity, never inferred proximity.
- Preserve unknown outcomes until evidence resolves them.
- Establish ownership, accepted scope, and readiness before automatic continuation.
- Keep launch, running execution, instruction acceptance, and completion distinct.
- Route coding work through the external harness boundary.
- Keep personal workspace and terminal policy in adapter configuration and code.

## Acceptance and delivery

The [overview](overview.md) defines internal work packages within one complete
implementation and direct trunk merge. Package 9 implements launch, inspection,
correlation, and recovery. Package 11 supplies the shared scheduler. Package 10
depends on both and adds bounded follow-through and controls.
Package 12 connects the full ambient/Activity experience. Each package verifies its
errors and lifecycle behavior before final integration.

Automated acceptance uses real task records, temporary storage, and fake external
adapters at the process boundary. Cover a successful launch, preparation failure,
unsupported capability, malformed receipt, timeout after actual launch, daemon
crash before receipt commit, and both hook/receipt orderings. Cover duplicate
continuation requests, changed scope, paused coordination, unknown readiness,
unrelated sessions, stop failure, and a late receipt after pause.

Run a real desktop acceptance with the personal adapter before declaring package 9's
integration complete. Verify the fresh worktree, visible native harness, loaded
normal configuration, actual hook correlation, session navigation, and task lookup
after daemon restart. Verify continuation and supported stop behavior for package 10.
Fake receipts alone do not establish that a real harness session is correlated.

## Alternatives and consequences

Embedding a coding harness in the internal SDK gives the pet implementation
capabilities outside its agreed role. Native desktop delegation preserves the
user's development environment and separates coordination from implementation.

Hard-coded Kitty or harness commands bind a public repository to one desktop
configuration. An external command protocol makes that configuration replaceable
while keeping operation identity and recovery testable in the core.

Terminal keystrokes without task lookup cannot establish restart-safe dispatch.
Adapters must account for their own delivery limits, and the pet must preserve
uncertainty when an underlying harness cannot prove an outcome.
