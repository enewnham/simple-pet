# Simple Pet

A native macOS desktop pet that speaks about your coding agents’ work as one character. It follows concurrent tasks across Claude Code, Codex, Cursor CLI, and command wrappers. A local Python daemon tracks the work and uses Claude Haiku for commentary. Swift/AppKit renders the pet.

## Quick start

Requires macOS 12+, Swift 5.9+, Git, [uv](https://docs.astral.sh/uv/), an Anthropic API key, and a pet sprite pack. uv installs Python 3.12 and the locked dependencies.

From the repository root, import your sprite ZIP and start the pet:

```sh
make setup
make build
scripts/fetch-assets.sh --zip ~/Downloads/pet.zip --pet my-pet
cp pet.config.example.json pet.config.json
export ANTHROPIC_API_KEY='your-api-key'
bin/pet-daemon
```

The imported sprites go into `assets/pets/my-pet`, matching the example configuration. For other import options and custom packs, see [fetch-assets.sh](scripts/fetch-assets.sh) and [import-assets.swift](scripts/import-assets.swift).

The pet floats across Spaces without a Dock icon. Drag it to reposition it or move it to another monitor. Click it to replay a message spoken within the last minute, or request fresh commentary. Long speech bubbles scroll. Ctrl-C or the right-click **Quit Pet** menu stops both the daemon and renderer.

During full idleness, the pet takes 20–45 second naps spaced 2–4 minutes apart. A small “Zzz” marks a nap. Packs with a `sleep` clip use that animation. Work, unresolved failures, speech, and mouse interaction keep it awake. Clicking wakes it immediately. Routine commentary pauses during naps.

Use `make run` to set up, build, and start subsequent runs.

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

Hooks forward goals, progress, selected command/result excerpts, and failures. Those excerpts are sent to Haiku for commentary. Hooks return without waiting for model output, and a missing daemon does not block the harness.

For commands outside a harness, use the wrapper. It preserves input, output, signals, and exit status:

```sh
examples/pet-run pytest
```

To request speech manually, run this in another terminal while the daemon is running:

```sh
pet-notify speak --context "The desktop pet is running"
```

The pet stays busy while any tracked tool call is active. See [pet-notify](bin/pet-notify) for manual session events and [pet-hook](scripts/pet-hook) for harness adapters.

## Customize

Edit `pet.config.json` to select sprites, choose a voice, adjust check-ins, or disable roaming:

```json
{
  "pet_dir": "assets/pets/my-pet",
  "persona": "You are a gentle, curious woodland pet who enjoys solving problems with a developer.",
  "idle_message_seconds": 300,
  "behavior": {
    "roaming": false
  }
}
```

Only `pet_dir` is required. The default persona is deadpan and slightly snarky, roaming is enabled, and check-ins occur after five minutes of quiet or waiting for input. Activity in another task does not reset that task's check-in timer.

Set `behavior.naps` to `false` to disable naps. `behavior.nap_interval` and `behavior.nap_seconds` accept minimum and maximum durations in seconds.

Paths are relative to the configuration file. Local configuration and imported assets are gitignored. Restart after changing the configuration. To use another file, run `bin/pet-daemon --config /absolute/path/to/pet.config.json`.

The pet remembers its conversation across restarts in `state/pet-memory.json`. Use a separate `memory_file` for each pet. To reset its memory, delete that file while the daemon is stopped.

See [config.py](pet_daemon/config.py) for all settings and defaults, and [cognition.py](pet_daemon/cognition.py) for the shared speech instructions.

## Development checks

Run the automated checks and compile the renderer:

```sh
make check
make build
```

The checks cover hooks, sockets, parallel tasks, sprite imports, and behavior without paid model requests. For a desktop smoke check in a logged-in macOS session, run `make desktop-check`. It uses the configured sprites and a fake API, so API credentials are unnecessary.

The daemon writes JSON Lines to `logs/pet-daemon.jsonl`. Logs contain task and command excerpts, model requests and responses, private thoughts, and behavior decisions. API keys and HTTP headers are excluded. Follow decisions with:

```sh
tail -f logs/pet-daemon.jsonl | jq 'select(.event == "observation.decision")'
```

For runtime details, see [daemon.py](pet_daemon/daemon.py), [sessions.py](pet_daemon/sessions.py), [behavior.py](pet_daemon/behavior.py), and [memory.py](pet_daemon/memory.py).

## License and attribution

Simple Pet is a fork and rebuild of [OpenPets](https://github.com/OpenPetsHQ/openpets). Credit goes to the OpenPets authors for the original project and artwork.

Simple Pet is licensed under the [MIT license](LICENSE), retaining OpenPets’ copyright notice and adding Elliot Newnham’s notice for modifications. The original license is preserved in [assets/LICENSE.OpenPets](assets/LICENSE.OpenPets).
