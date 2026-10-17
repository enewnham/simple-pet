# Simple Pet

A native macOS desktop pet that speaks about your coding agents’ work. A local Python daemon tracks tasks and uses Claude Haiku for commentary. Swift/AppKit renders the pet. Purposefully simplified and opinonated.

![Simple Pet demo](docs/media/simple-pet.webp)

- Tracks concurrent work across Claude Code, Codex, Cursor CLI, and command wrappers.
- Speaks in a configurable persona and remembers work and recent remarks across restarts.
- Roams, reacts to work, and takes short naps when idle.
- Supports custom sprite packs, dragging, and clicking interactions.

Future plans

- Engage with meeting transcripts and slack conversations
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

Console logs show timestamps, levels, session lifecycles, and commentary outcomes with
timing and token usage. Use `bin/pet-daemon --debug` for scheduling and decision logs,
or `bin/pet-daemon --verbose` for raw hooks, renderer packets, and API payloads.
Levels use color in a terminal. Set `NO_COLOR=1` to disable color.

The `log_file` setting controls the rotating JSONL log, which defaults to
`logs/pet-daemon.jsonl`. It retains all log levels, full model requests and responses,
and renderer packets. Observation IDs connect requests to their decisions and delivery.
Session logs identify alias merges, ignored hooks, stale turns, and retention removal
to help diagnose sessions that remain after an agent exits.

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

Customize `persona` to change the pet's voice. Use `memory_retention_days` and
`memory_max_events` to limit remembered work. Install and authenticate `gh` to include
PR status in commentary.

Clear speech or remembered work with:

```sh
bin/pet-notify clear-speech  # Forget recent remarks and dismiss the bubble
bin/pet-notify clear-memory  # Clear remembered work and speech
```

Set `PET_DAEMON_SOCKET` if the daemon uses a custom socket path.

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
