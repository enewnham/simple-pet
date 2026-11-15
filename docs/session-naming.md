# Session naming callback

Naming is disabled by default. Enable it with a `naming` block in
`pet.config.json`, then restart the daemon with `make restart-agent`:

```json
{
  "naming": {
    "command": ["/absolute/path/to/naming-command"],
    "debounce_seconds": 10,
    "timeout_seconds": 10
  }
}
```

Simple Pet derives names locally without a model API call. It uses the effective
workstream `title`, then `goal`, then the session's first observed intent prompt.
It removes embedded instructions and code blocks, takes the first nonempty prose
line, strips common request prefixes, and limits the result to eight words and
64 characters. Harness and terminal identifiers never supply name text. This is
text shortening rather than semantic summarization. A workstream title gives
more precise names when a prompt begins with general setup prose.

The daemon checks workstream marks once per second for up to 1,000 observed
sessions. Each changed proposed name waits for `debounce_seconds` without another
name change before delivery. Further prompts preserve the first prompt. A mark
can rename a session without another hook arriving. Removing a title or goal
returns to the next available source.

The command receives one JSON object on stdin:

```json
{
  "harness": "codex",
  "session": "native-session-id",
  "thread_id": "native-session-id",
  "zmx_session": "terminal-session",
  "cwd": "/path/to/checkout",
  "name": "Fix socket timeout"
}
```

`thread_id` is the harness's native session identifier, also exposed as `session`.
Missing zmx and cwd values are `null`. Metadata supplied by later hooks updates
the remembered session. Names use the notification text redactor before shortening.

The argv command executes directly with the fixed system `PATH` used by
[notifications](notifications.md). Stdout and stderr are discarded. Timeout and
shutdown kill the command's process group. A failed command is logged without the
payload and is not retried. The same proposed name does not trigger another
callback. Naming state lives in memory, so a daemon restart can deliver a name
again after observing a session. Removing the `naming` block disables naming.

## zmx example

The installed zmx CLI documents `zmx set <session> k=v ...` for session labels.
[examples/name-zmx-session.py](../examples/name-zmx-session.py) sets the `title`
label. Supply absolute paths to Python, the example, and your zmx executable:

```json
{
  "naming": {
    "command": [
      "/usr/bin/python3",
      "/absolute/path/to/simple-pet/examples/name-zmx-session.py",
      "/opt/homebrew/bin/zmx"
    ]
  }
}
```

The example skips sessions without a zmx identifier and passes the name as one
argument. It needs no dotfiles, agent session database, or agent management tool.
