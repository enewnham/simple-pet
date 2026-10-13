# Simple Pet

A native macOS desktop pet that speaks about your coding agents’ work. A local Python daemon tracks tasks and uses Claude Haiku for commentary. Swift/AppKit renders the pet. Purposefully simplified and opinonated.

- Tracks concurrent work across Claude Code, Codex, Cursor CLI, and command wrappers.
- Speaks in a configurable persona and remembers conversations across restarts.
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

Hooks send task and command excerpts to Haiku for commentary.

For commands outside a harness, use the wrapper:

```sh
examples/pet-run pytest
```

## Configuration

Edit `pet.config.json` and restart the daemon to apply changes. Paths are relative to the configuration file. See [pet.config.example.json](pet.config.example.json) for all settings and defaults. Set `pet_dir` to your imported sprite pack.

Conversation exchanges and summaries are saved to `memory_file`. Claude requests use a one-hour prompt cache. During active work, `summary_interval` controls how many older exchanges become a memory while recent exchanges remain in the conversation. After `idle_compaction_seconds` without a conversation exchange (55 minutes by default), all remaining exchanges become a memory. Idle memory compaction produces no speech, and routine observations run only while a task is active. `summary_summary_interval` controls consolidation of older memories.

Computer sleep pauses API requests and cancels in-flight requests. Completed exchanges remain saved. After wake or a daemon restart, elapsed wall time determines whether the conversation needs compaction before its next observation. Memory compaction runs after wake when due, even if the cloud cache has expired. The pet's short nap animations follow their own behavior settings.

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
