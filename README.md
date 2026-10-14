# Simple Pet

A native macOS desktop pet that speaks about your coding agents’ work. A local Python daemon tracks tasks and uses Claude Haiku for commentary. Swift/AppKit renders the pet. Purposefully simplified and opinonated.

- Tracks concurrent work across Claude Code, Codex, Cursor CLI, and command wrappers.
- Speaks in a configurable persona and remembers work and recent remarks across restarts.
- Roams, reacts to work, and takes short naps when idle.
- Supports custom sprite packs, dragging, and clicking interactions.

Future plans

- Engage with meeting transcripts and slack conversations
- Tomogatchi-style care and feeding (with tokens)
- Open to community contributions

## Quick start

Requires macOS 12+, Swift 5.9+, Git, [uv](https://docs.astral.sh/uv/), an Anthropic API key, and a pet sprite pack.

Find pets at [OpenPets packs](https://openpets.dev/packs) or [OpenPet Hatchery](https://github.com/chenmi319/openpet-hatchery). Download a pet ZIP, then run these commands from the repository root:

```sh
make setup
make build
scripts/fetch-assets.sh --zip ~/Downloads/pet.zip --pet my-pet
cp pet.config.example.json pet.config.json
export ANTHROPIC_API_KEY='your-api-key'
bin/pet-daemon
```

The daemon runs in the foreground. Press Ctrl-C to stop it. Use `make run` for subsequent runs.

## Connect your agents

Add the client and hook adapter to PATH before launching your harness:

```sh
export PATH="$PWD/bin:$PWD/scripts:$PATH"
```

Merge the hooks from the matching example into your harness configuration, preserving unrelated settings and handlers:

| Harness | Example | Destination |
| --- | --- | --- |
| Claude Code | [claude-settings.json](examples/claude-settings.json) | `~/.claude/settings.json` |
| Codex | [codex-hooks.json](examples/codex-hooks.json) | `~/.codex/hooks.json` |
| Cursor CLI | [cursor-hooks.json](examples/cursor-hooks.json) | `~/.cursor/hooks.json` |

If the harness cannot find `pet-hook` through PATH, replace it in the hook commands with the absolute path to `scripts/pet-hook`. In Codex, review and trust the definitions through `/hooks`.

Hooks send task and command metadata to Haiku for commentary. Failures describe broad
activity and outcome, such as tests failing or a build succeeding. Tool output, stack traces,
and diagnostic details are discarded.

For commands outside a harness, use the wrapper:

```sh
examples/pet-run pytest
```

## Configuration

Edit `pet.config.json` and restart the daemon to apply changes. Paths are relative to the configuration file. See [pet.config.example.json](pet.config.example.json) for all settings and defaults. Set `pet_dir` to your imported sprite pack.

Factual events, project context, and dispatched speech are saved in SQLite at `memory_file`.
Each session keeps its first captured prompt, four recent requests, and a short evolving
task summary. Haiku can update the focused session's summary in its speech response.
The summary describes current intent, while the first prompt remains a historical anchor.
These fields survive restarts without restoring live calls or activity. Stale responses
do not overwrite newer session context.

Each request includes bounded current session state, five recent events, three dispatched
remarks, and at most two relevant older events from the same project directory. Historical
lookup uses explicit task goals and SQLite FTS5. Stored memory does not establish live
activity after a restart. Memory schema changes discard stored data without migration.

`prompt_max_bytes` caps the serialized request, including the persona and response schema.
It is a byte limit, not an exact token limit. `memory_retention_days` and `memory_max_events`
limit stored history. The daemon preserves the triggering session when trimming context.
Persona text must fit within 4,000 UTF-8 bytes. Requests use a one-hour cache for the stable
system prompt and a concise structured response.

The pet reacts to starts, results, meaningful progress, new failures, and prolonged waits.
Unchanged routine observations make no API call. Repeated failures in the same activity earn reactions
on occurrences one, three, and five. A successful execution of that activity clears its
failure count. Unrelated commands do not clear it. Clicks can replay recent speech or request a fresh
comment. Explicit commentary requests and periodic check-ins remain available.

Computer sleep cancels in-flight requests and pauses generation. Wake discards queued
reactions and rechecks stale sessions. Factual events remain stored. Only nonempty speech
successfully sent to the renderer with a nonzero lifetime enters speech memory. Socket
success records dispatch, not confirmation that the renderer displayed the bubble.

## Development checks

Run the automated checks and compile the renderer:

```sh
make check
make build
```

Run `make desktop-check` in a logged-in macOS session to test the renderer. Checks use a fake API and make no paid model requests.

## License and attribution

Simple Pet is a fork and rebuild of [OpenPets](https://github.com/OpenPetsHQ/openpets). Credit goes to the OpenPets authors for the original project and artwork.

Simple Pet is licensed under the [MIT license](LICENSE), retaining OpenPets’ copyright notice.
