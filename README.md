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

The pet remembers recent work and remarks across restarts. Use `memory_retention_days`
and `memory_max_events` to limit how much history it keeps. Customize `persona` to change
its voice.

Session context is summarized on each user prompt. Set `summarize_on` to an array of
commentary reasons: `prompt`, `completion`, `routine`, `failure`, `check_in`, `click`,
or `explicit`. An empty array disables summaries. Set `summarize_first_request_only`
to `true` to override this array and summarize only the initial prompt of each session.
The initial summary gets one attempt, including across daemon restarts.

Prompt excerpts retain up to 1,000 characters within the hook packet's 1,999-byte limit.
Long prompts use sentence ranking to prioritize requests and questions, preserving
selected sentences in their original order and marking omitted text. Excerpts can
miss relevant instructions. Summaries use the configured remote model.

The pet thinks before starting work, celebrates successes, and reacts to failures and
long waits. After a quiet stretch, it may wave for attention. Click it to hear a recent
remark or get a fresh response. After extended idleness, a click also earns a greeting.
Set `behavior.idle_greeting_seconds` to adjust how long it waits before greeting.

Clear the running pet's memory with:

```sh
bin/pet-notify clear-memory
```

This clears remembered work, remarks, and the current speech bubble. The pet learns from
new agent activity. Set `PET_DAEMON_SOCKET` if it uses a custom socket path.

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
