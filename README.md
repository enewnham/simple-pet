# Simple Pet

A native macOS desktop pet that speaks about your coding agents’ work. A local Python daemon tracks tasks and uses Claude Haiku for commentary. Swift/AppKit renders the pet. Purposefully simplified and opinonated.

![Simple Pet demo](docs/media/simple-pet.webp)

- Tracks concurrent work across Claude Code, Codex, Cursor CLI, and command wrappers.
- Speaks in a configurable persona and remembers work across restarts.
- Roams, reacts to work, and takes short naps when idle.
- Supports custom sprite packs, dragging, and clicking interactions.

Future plans

- Engage with meeting transcripts and slack conversations
- Open to community contributions

## Quick start

Requires macOS 12+, Swift 6.0+, Git, [uv](https://docs.astral.sh/uv/), an Anthropic API key, and a pet sprite pack.

Find pets at [OpenPets packs](https://openpets.dev/packs) or [OpenPet Hatchery](https://github.com/chenmi319/openpet-hatchery). Sprite sheets must use the V2 1536×2288 layout. Custom manifests and PNG sprite folders are also supported. Download a pet ZIP, then run these commands from the repository root:

```sh
make setup
make build
scripts/fetch-assets.sh --zip ~/Downloads/pet.zip --pet my-pet
cp pet.config.example.json pet.config.json
export ANTHROPIC_API_KEY='your-api-key'
bin/pet-daemon
```

The daemon runs in the foreground. Press Ctrl-C to stop it. Use `make run` for subsequent runs.

Console logs go to stderr with local timestamps, colored levels, event names, and
key-value fields. Colors are disabled when stderr is redirected or `NO_COLOR` is set.

- `bin/pet-daemon` shows lifecycle events, agent activity, commentary text, and warnings.
- `bin/pet-daemon --debug` adds decisions and timings while hiding bulky payloads.
- `bin/pet-daemon --verbose` includes full model requests, responses, and renderer packets.

Every mode writes all levels and full payloads to `logs/pet-daemon.jsonl`, or the
configured `log_file`. Files rotate at 10 MiB with three backups. JSON records have
UTC timestamps, levels, event names, logger names, and a process run ID. Commentary
requests also carry an observation ID, reason, subject, and session across async calls.
Model payloads can contain user prompts and agent output.

Typical console output looks like:

```text
10:42:01 I daemon.listening             model=claude-haiku-5-5 pet=my-pet socket=/tmp/pet-daemon.sock
10:42:06 I observation.finished         duration_seconds=0.842 observation_id=1a2b3c4d outcome=submitted reason=check_in
10:42:06 I speech.delivery              outcome=accepted text='The parser tests caught that edge case.'
```

To inspect one request, use its full observation ID from the JSON file:

```sh
jq 'select(.observation_id == "FULL_OBSERVATION_ID")' logs/pet-daemon.jsonl
```

Browse commentary requests in a terminal:

```sh
bin/pet-inspect
bin/pet-inspect /path/to/pet-daemon.jsonl
```

The viewer shows a timeline of commentary attempts beside the speech bubble,
system prompt, and messages sent to Claude. Markdown is rendered and JSON is
indented. Rows show delivery outcomes, including displayed and silent attempts.
Story-generation requests are excluded. The viewer reads the log and its numbered
rotations. Use ↑/↓ to select a request, Tab to move between panes, the mouse wheel
to scroll, r to refresh, and q to quit.

### Try saved work at a faster pace

Run a second pet with your saved work, configured model, persona, and sprite pack:

```sh
make demo
# Focus a saved thread and set the pace:
make demo ARGS='--work "parser" --speed 10'
```

`--work` matches request or report text. Omit it to choose the most recent recorded
thread. The demo uses `ANTHROPIC_API_KEY`, copies local memory to a temporary
directory, and uses private sockets. The live pet can keep running. Timing advances
at 10× speed by default, while reported facts retain their recorded timestamps.
Click the pet and try its offered choices, or use Recent work for a tour. Restarting
the demo resets its feedback to the saved copy. Press Ctrl-C to close it.

## Connect your agents

Add the client and hook adapter to PATH before launching your coding tool:

```sh
export PATH="$PWD/bin:$PWD/scripts:$PATH"
```

Merge the hooks from the matching example into your coding tool configuration, preserving unrelated settings and handlers:

| Coding tool | Example | Destination |
| --- | --- | --- |
| Claude Code | [claude-settings.json](examples/claude-settings.json) | `~/.claude/settings.json` |
| Codex | [codex-hooks.json](examples/codex-hooks.json) | `~/.codex/hooks.json` |
| Cursor CLI | [cursor-hooks.json](examples/cursor-hooks.json) | `~/.cursor/hooks.json` |

If the coding tool cannot find `pet-hook` through PATH, replace it in the hook commands with the absolute path to `scripts/pet-hook`. In Codex, review and trust the definitions through `/hooks`.

Hooks provide agent intent, reported progress, and execution signals. The pet uses
execution signals for activity and animation. Tool output, stack traces, and diagnostic
details are discarded.

For commands outside a coding tool, use the wrapper:

```sh
examples/pet-run pytest
```

## Configuration

Edit `pet.config.json` and restart the daemon to apply changes. Paths are relative to the configuration file. See [pet.config.example.json](pet.config.example.json) for all settings and defaults. Set `pet_dir` to your imported sprite pack.

Customize `persona` to change the pet's voice. Use `memory_retention_days` and
`memory_max_events` to limit remembered work. Install and authenticate `gh` for
PR links, stack membership, and basic PR state. Repository observations provide context
for the agents' reports.

The pet maintains a short work story from user requests and scoped agent reports before
generating commentary. Quoted text, code blocks, and tables are excluded from reports.
Recent pet remarks stay local for repetition checks and are excluded from model prompts.
Related sessions can draw on remembered decisions and reports. The pet reacts to recent
results and live work, and can recall where we left off during quiet periods or an idle
click. Recollections grow less frequent as their evidence ages. Finished work recedes as
the conversation moves on.

Automatic activity reactions stay brief. Clicking the pet opens a conversation,
including while agents are working. A separate prompt composes its headline,
remark, context, and up to two relevant responses from available actions.
**Recent work** offers another thread or a brief tour of recent work and explicitly
remaining threads. The right-click menu also offers rounds.

Work commentary offers actions for its recorded worktree, PR, or session when a
handler is configured. When offered, **Already handled** stops recalling that source,
and **Tomorrow** defers it until 09:00 the next local day. Related sessions share
feedback on the same source. Opening a destination briefly spaces out further
recalls. A failed open leaves the action available to retry.

`navigation.handlers` configures destination handlers. Worktrees and PRs use
`/usr/bin/open` by default. Set a handler to `null` to hide its action. Sessions
require an explicit handler and use their recorded source and native session ID:

```json
"navigation": {
  "handlers": {
    "worktree": {"uri_template": "vscode://file{path}"},
    "pull_request": {"argv": ["/usr/bin/open", "{target}"]},
    "session": null,
    "timeout_seconds": 10
  }
}
```

This example opens checkouts in VS Code. A session handler must use the URI or
executable supported by your session host.

Each handler specifies either `uri_template` or `argv`. URI placeholders are
percent encoded. `{path}` preserves directory separators for file URLs. Use
`{target}` for a value inside a URI query. Other placeholders are `{session_id}`
and `{source}`. Executable arguments substitute these
values literally and receive the target as JSON on standard input. Executables
run directly without a shell. The pet reports failed or timed-out opens.

Remembered work and speech are stored locally. Use the commands below to clear them.

Use `prompt_target_bytes` to control how much optional context is included in model
requests. Requests can exceed the target to preserve complete context.

The pet occasionally reacts to negative user prompts with its sad animation.
Set `behavior.sentiment_reactions` to `false` to disable these reactions.

Clear speech or remembered work with:

```sh
bin/pet-notify clear-speech  # Forget recent remarks and dismiss the bubble
bin/pet-notify clear-memory  # Clear remembered work and speech
```

Set `PET_DAEMON_SOCKET` if the daemon uses a custom socket path.

## Development checks

Run checks

```sh
make check
```

Replay commentary fixtures without API calls:

```sh
uv run --locked python -m pet_daemon.replay
```

Pass a fixture file or directory to select sequences. To generate a model sample
per sequence using the configured persona and model:

```sh
uv run --locked python -m pet_daemon.replay --model-run --config pet.config.json
```

Results, requests, latency, and token usage are saved under `.cache/replays/`.
Use `--samples 3` for repeated samples. Review the comparisons before committing prompt changes.

Use `$pet-triage` in Codex to investigate recent commentary, reactions, movement,
naps, or timing. Describe the behavior or paste a log entry. The
[skill](.agents/skills/pet-triage/SKILL.md) saves Markdown findings, evidence, and
replay comparisons under `.cache/triage/`. Follow-up prompts continue the same case.

## License and attribution

Simple Pet is a fork and rebuild of [OpenPets](https://github.com/OpenPetsHQ/openpets). Credit goes to the OpenPets authors for the original project and artwork.

Simple Pet is licensed under the [MIT license](LICENSE), retaining OpenPets’ copyright notice.
