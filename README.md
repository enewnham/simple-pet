# Simple Pet

A native macOS desktop pet that speaks about your coding agents’ work. A local Python daemon tracks tasks and uses Claude Haiku for commentary. Swift/AppKit renders the pet. Purposefully simplified and opinonated.

![Simple Pet demo](docs/media/simple-pet.webp)

- Tracks concurrent work across Claude Code, Codex, Cursor CLI, and command wrappers.
- Speaks in a configurable persona and remembers session context and recent remarks across restarts.
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

Use `bin/pet-daemon --debug` or `--verbose` for troubleshooting. Logs are also available at `logs/pet-daemon.jsonl`.

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

Hooks provide agent intent, reported progress, and execution signals. The pet uses
execution signals for activity and animation. Tool output, stack traces, and diagnostic
details are discarded.

For commands outside a harness, use the wrapper:

```sh
examples/pet-run pytest
```

## Configuration

Edit `pet.config.json` and restart the daemon to apply changes. Paths are relative to the configuration file. See [pet.config.example.json](pet.config.example.json) for all settings and defaults. Set `pet_dir` to your imported sprite pack.

Customize `persona` to change the pet's voice. Use `memory_retention_days` and
`memory_max_events` to limit remembered work. Install and authenticate `gh` for
PR links, stack membership, and basic PR state. Repository observations provide context
for the agents' reports.

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

Run the checks in a logged-in macOS session:

```sh
make check
```

`make setup` installs runtime and development dependencies from `uv.lock`.
`make check` includes Ruff linting, Ruff format verification, and ty type checking
for all Python modules, tests, and Python scripts. Ruff extends its default rules
with required function annotations. ty requires type arguments for generic types.
Both tools infer the Python version from `pyproject.toml`.
Run `make lint` for just these checks or `make format` to apply formatting.

Use dataclasses and field access for internal records. Reuse SDK types at API
boundaries. Keep `TypedDict` for JSON payloads and typed keyword arguments, and
validate external data when decoding it into internal records. Use standard
dataclass helpers and explicit JSON codecs for serialization.

`make test` includes renderer checks. Checks use a fake API and make no paid model requests.

## License and attribution

Simple Pet is a fork and rebuild of [OpenPets](https://github.com/OpenPetsHQ/openpets). Credit goes to the OpenPets authors for the original project and artwork.

Simple Pet is licensed under the [MIT license](LICENSE), retaining OpenPets’ copyright notice.
