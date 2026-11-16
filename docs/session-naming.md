# Session naming

Naming is disabled by default. Enable it with a `naming` block in
`pet.config.json`, then restart the daemon with `make restart-agent`:

```json
{
  "naming": {
    "command": ["/absolute/path/to/naming-command"],
    "timeout_seconds": 10
  }
}
```

At the first `intent` of a top-level session with a `zmx_session`, Simple Pet
immediately asks its configured Anthropic model to name the work. Sessions with
a parent and sessions without a zmx identifier are ignored. Later prompts and
workstream mark changes do not trigger naming.

The model receives the first prompt through `report_prose`, bounded to 4,000
characters, plus the checkout repository path, branch, pull request, and workstream
mark title and goal when set. The existing notification redactor processes all
input before the request. The model uses the configured model and request timeout.
The prompt asks for a short name of about 12 characters. The response schema
requires one to three plain words containing lowercase letters and digits,
separated by single spaces. Examples include `naming design`, `wm 730 adr`, and
`raycast reset`. Hyphens are rejected. Length is a target, with no enforced limit.
Names are delivered exactly as validated.

The `session_names` table records the attempt before the model request and stores
a valid name before delivery. A restart preserves the attempt. A model error,
timeout, or invalid output logs `naming.failed` and leaves the session unnamed.
Each session gets one attempt, with no retries or renames.

The command receives one JSON object on stdin:

```json
{
  "harness": "codex",
  "session": "native-session-id",
  "thread_id": "native-session-id",
  "zmx_session": "terminal-session",
  "cwd": "/path/to/checkout",
  "name": "socket fix"
}
```

`thread_id` is the harness's native session identifier, also exposed as `session`.
A missing cwd is `null`. Simple Pet supplies this contract without depending on
an agent management tool.

The argv command executes directly with the fixed system `PATH` used by
[notifications](notifications.md). Stdout and stderr are discarded. The callback
uses `timeout_seconds`. Timeout and shutdown kill the command's process group.
A failed callback is logged and is not retried. Removing the `naming` block
disables naming.
