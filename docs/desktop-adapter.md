# Desktop adapters

Simple Pet delegates coding to a configured desktop adapter. The adapter prepares a
workspace and starts a native harness with its normal configuration. It owns the
terminal, worktree, bootstrap commands, and harness-specific lifecycle operations.
The pet stores task intent, accepted authority, operation receipts, and observations.

Configure the adapter in the `desktop` section of `pet.config.json`:

```json
{
  "desktop": {
    "name": "my-desktop",
    "argv": ["/path/to/desktop-adapter"],
    "capabilities": ["launch", "inspect", "continue", "open", "stop"],
    "timeout_seconds": 30
  }
}
```

The executable receives one JSON request on standard input and writes one JSON
result to standard output. Send diagnostics to standard error. The pet invokes the
argument vector directly. Store credentials and personal commands in your adapter's
configuration.

Launch requires inspection support. Continuation, navigation, and stop are optional.
A responsibility must explicitly permit `launch` or `continue` before the pet can
dispatch those operations. Repository paths must fall within its adopted scope.

## Requests

Every request uses protocol version 1 and identifies its task and command. The pet
persists the command before invoking the executable. This launch request illustrates
the fields:

```json
{
  "protocol_version": 1,
  "operation": "launch",
  "command_id": "a17eec2d-36ce-454e-8653-473a8cf5126f",
  "task_id": "31ef5b97-5542-4b38-8897-b58e37941120",
  "scope_revision": 1,
  "instruction_revision": 2,
  "task_revision": 1,
  "user_revision": 0,
  "expected_interaction_revision": null,
  "instruction": "",
  "expected_status": "preparing",
  "expected_readiness": "unknown",
  "brief": {
    "objective": "Validate the migration rollback",
    "expected_result": "A validation report with supporting check evidence",
    "repository": "/work/project",
    "constraints": ["Do not roll out the migration"],
    "sources": []
  },
  "session": null,
  "environment": {
    "SIMPLE_PET_TASK_ID": "31ef5b97-5542-4b38-8897-b58e37941120",
    "SIMPLE_PET_COMMAND_ID": "a17eec2d-36ce-454e-8653-473a8cf5126f"
  }
}
```

Pass the supplied environment values into the launched harness. The `pet-hook`
entrypoint carries them into native hook events. The daemon correlates an accepted
hook only with an existing task and launch or continuation command. A matching
checkout, parent process, branch, or pull request does not establish ownership.

For `continue`, supply the bounded `instruction` to the established `session` only
when the native harness still satisfies the expected status and readiness. Preserve
the native session locator in each receipt. A prepared terminal paste establishes a
handoff only when the harness confirms that it accepted the instruction.

For `inspect`, look up the supplied command and task. An inspection of an unresolved
launch or continuation retains that operation's command ID. A general task refresh
uses a separate inspection command. Inspection must not launch work or resubmit an
instruction.

For `open`, navigate to the established session. For `stop`, request termination of
that session. A stop request has its own command ID. Report stopped execution only
when it is observed.

## Results

Return the same task and command identity. A receipt can establish a preparing
workspace before native hooks arrive:

```json
{
  "protocol_version": 1,
  "task_id": "31ef5b97-5542-4b38-8897-b58e37941120",
  "command_id": "a17eec2d-36ce-454e-8653-473a8cf5126f",
  "outcome": "accepted",
  "observed_at": 1793898000,
  "status": "preparing",
  "readiness": "unknown",
  "interaction_revision": null,
  "external_id": "desktop-task-42",
  "session": {
    "adapter": "my-desktop",
    "harness": "codex",
    "session": "native-session-42",
    "workspace": "/work/project-validation",
    "external_id": "desktop-task-42"
  },
  "error": null
}
```

`outcome` is `pending`, `accepted`, `failed`, or `unknown`. It describes the operation.
`status` describes observed execution: `preparing`, `running`, `ready`, `waiting`,
`ended`, `stopped`, or `unknown`. `readiness` is `ready`, `busy`, `waiting`, or
`unknown`. `observed_at` is the observation's Unix wall time. Missing status or
readiness cannot authorize a continuation.

For automatic continuation, inspection must also return a nonempty
`interaction_revision` derived from native input or execution evidence. Check the
request's `expected_interaction_revision` against the harness immediately before
delivering an instruction. Return a failed or unknown outcome when it changed.
A terminal being open or a process being alive does not establish readiness or
this revision. An adapter that cannot establish both supports observation and a
prepared human handoff without automatic continuation.

An ended execution or an accepted command does not establish that the responsibility
is complete. The coordinator evaluates the task's expected result against evidence.
A human direction changes the task's user revision and invalidates conflicting
queued instructions. Pausing coordination prevents proactive launches and
continuations. A running harness remains available for inspection and a separate
stop request.

## Recovery

Retain task and command mappings across adapter restarts. Deduplicate every mutating
request by its command ID. Persist enough information to inspect an operation whose
response was lost after the harness accepted it. A timeout, malformed result, or
lost connection leaves the outcome unknown. The pet inspects the original command
before considering another instruction.

If a hook arrives before its receipt, preserve the explicit native association.
A later receipt can establish that the session exists after a responsibility is
paused. It does not authorize another continuation. A schema reset clears local
authority and does not adopt surviving external sessions.

The automated process fixture in
[`Tests/fixtures/desktop/adapter.py`](../Tests/fixtures/desktop/adapter.py) exercises
receipts, timeouts, inspection, and native hook correlation. It simulates execution
and does not launch a coding harness. Real desktop acceptance must separately verify
workspace creation, visible native execution, hook association, navigation, recovery,
and any supported continuation or stop operation.
