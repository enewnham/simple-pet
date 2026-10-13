# Simple Pet

A native macOS desktop pet driven by Claude Code, Codex, Cursor CLI, and command wrappers. A local Python daemon tracks parallel tool calls, schedules sprite behavior, and maintains a small Haiku conversation. The Swift/AppKit renderer displays the configured pet.

## Build and start

Requires macOS 12+, Swift 5.9+, Git, and [uv](https://docs.astral.sh/uv/). uv installs the Python 3.12 environment and locked Anthropic SDK dependencies.

```sh
make setup
make build
scripts/fetch-assets.sh
export ANTHROPIC_API_KEY='your-api-key'
bin/pet-daemon
```

The daemon launches its renderer and stays in the foreground. Ctrl-C or the pet's right-click **Quit Pet** menu stops both processes. `make run` performs setup, builds, and starts the daemon.

The 200×160 transparent panel starts in the bottom-right corner, floats across Spaces, and has no Dock icon. Drag the sprite to reposition it. Walking stays on its current monitor. Dragging it onto another monitor changes the walking area.

Startup requires API credentials and a valid sprite pack. Runtime API errors keep hook delivery and sprite behavior running. The daemon owns its renderer process and removes its own socket on shutdown. A live listener at either configured socket is left intact and startup fails.

## Configure the pet

Edit [pet.config.json](pet.config.json), then start the daemon. Select a sprite pack through `pet_dir`. Paths are relative to the configuration file.

```sh
bin/pet-daemon --config /absolute/path/to/pet.config.json
```

Configuration is loaded once. Pet selection stays fixed for the run. Restart after changing the pet or configuration.

| Setting | Default |
| --- | --- |
| `pet_dir` | `assets/pets/builtin` |
| `overlay_binary` | `.build/pet-overlay` |
| `daemon_socket` | `/tmp/pet-daemon.sock` |
| `overlay_socket` | `/tmp/pet.sock` |
| `model` | `claude-haiku-5-5` |
| `observation_seconds` | Random interval between 60 and 90 |
| `history_pairs` | 8 complete user/assistant pairs |
| `speech_ttl` | 5 seconds |
| `request_timeout` | 20 seconds |
| `stale_seconds` | 3600 seconds |
| `completed_retention_seconds` | 7200 seconds |
| `max_tokens` | 4096 API output tokens |

`behavior` accepts these settings:

| Setting | Default |
| --- | --- |
| `roaming` | `true` |
| `repeat_range` | `[1, 3]` complete cycles |
| `repeat_budget` | 4 seconds, with longer clips played once |
| `gesture_cooldown` | 15 seconds after a gesture |
| `baseline_dwell` / `idle_debounce` | 2 / 1 seconds |
| `error_seconds` | 2 seconds |
| `walk_seconds` / `walk_speed` | `[2, 6]` seconds / 40 pixels per second |
| `idle_pause` / `working_pause` | `[20, 60]` / `[8, 20]` seconds |
| `work_burst_seconds` | `[2, 4]` seconds |
| `pointer_radius` / `resume_seconds` | 100 pixels / 3 seconds |
| `gaze_reset_seconds` / `gaze_deadzone` | 1.2 seconds / 24 pixels |

Observation timing and behavior timing are independent. The daemon chooses walking direction, targets, repeats, and pauses. Haiku suggests optional gestures using the selected pet's available clips.

Speech frequency, brevity, and restrained snark are guided by the system prompt in [cognition.py](pet_daemon/cognition.py). The daemon accepts nullable speech and applies no word-count limit, prose trimming, silence lottery, or automatic speech cooldown. `max_tokens` supplies the Anthropic API's required output budget. Socket payloads still have transport limits.

## Try parallel sessions

Add the client and hook adapter to PATH before launching harnesses:

```sh
export PATH="$PWD/bin:$PWD/scripts:$PATH"
```

Use another terminal while the daemon runs:

```sh
pet-notify intent --session demo-a --text "Refactor batch inserts"
pet-notify --session demo-a --status start --tool "cargo test"
pet-notify --session demo-b --status start --tool "pytest"

pet-notify --session demo-a --status done
# demo-b is still working.

pet-notify --session demo-b --status error --detail "AssertionError in parser tests"
pet-notify speak --context "One test suite passed; the parser suite failed"
```

For overlapping tools in one manual session, supply a different `--call` for each operation and repeat it on completion:

```sh
pet-notify --session demo --call build --status start --tool "cargo build"
pet-notify --session demo --call tests --status start --tool "cargo test"
pet-notify --session demo --call build --status done
pet-notify --session demo --call tests --status done
```

`speak` returns after queuing a request. It uses the daemon's conversation and explicitly requests speech. Optional `--ttl` changes that bubble's expiry. Hooks and the client never wait for model output.

All tool events count pending calls. The pet reports `working` while any remain and `idle` otherwise, including reasoning and human-input gaps. Errors preempt ordinary behavior briefly. Turn completion is tracked separately, so a tool gap does not count as pipeline convergence. Cancelled and stale work do not count as clean completion.

Sessions use native identifiers and timestamps. Stale lanes expire after the configured interval. Conversation and session history stay in memory. A restart begins a fresh history.

## Native hooks

Configure each harness using its example:

| Harness | Example | Destination |
| --- | --- | --- |
| Claude Code | [claude-settings.json](examples/claude-settings.json) | `~/.claude/settings.json` |
| Codex | [codex-hooks.json](examples/codex-hooks.json) | `~/.codex/hooks.json` |
| Cursor CLI | [cursor-hooks.json](examples/cursor-hooks.json) | `~/.cursor/hooks.json` |

Merge hook arrays without removing unrelated settings or handlers. Replace `pet-hook` in the commands with the absolute path to `scripts/pet-hook`, or use the PATH setup above. In Codex, review and trust the definitions through `/hooks`.

Adapters forward prompt intent, selected command/result excerpts, progress, failures, and native tool/session/subagent identifiers. They do not read full transcripts or make API calls. Native inputs are untrusted data in the persona prompt. Excerpts forwarded to Haiku may contain task or command details.

Hook responses preserve harness decisions: `{}` normally and `{"continue":true}` for Cursor prompt submission. Cursor's imported Claude hook payloads are skipped to avoid duplicate pet events.

Claude terminal failures close the turn and trigger an alert. Codex failures use structured tool result fields. Cursor failures and cancellation statuses are normalized for tools, subagents, and turns. Subagent aliases and parent relationships keep parallel work correlated.

See the official [Claude hooks](https://code.claude.com/docs/en/hooks), [Codex hooks](https://developers.openai.com/codex/hooks), and [Cursor hooks](https://cursor.com/docs/hooks) references.

## Wrap CLI commands

The wrapper forwards command intent and lifecycle while preserving standard input, output, signals, and exit status:

```sh
examples/pet-run pytest
examples/pet-run cargo build
examples/pet-run git commit -m 'Add parser'
printf 'hello\n' | examples/pet-run cat
```

It uses a unique session for each invocation. Failures automatically schedule a daemon observation. Persona judgment determines whether an automatic observation speaks.

## Sprite packs

The fetch script imports existing artwork from [OpenPets](https://github.com/OpenPetsHQ/openpets):

```sh
scripts/fetch-assets.sh --pet builtin
scripts/fetch-assets.sh --pet NAME --source /path/to/openpets
scripts/fetch-assets.sh --source /path/to/custom-pack --pet custom
scripts/fetch-assets.sh --zip ~/Downloads/pet.zip
scripts/fetch-assets.sh --zip ~/Downloads/pet.zip --pet my-pet
scripts/fetch-assets.sh --ref COMMIT
```

ZIP files can contain the pet files at the archive root or inside one top-level folder. The default name comes from `pet.json` (`id`), `manifest.json` (`pet`), or the ZIP filename. Use `--pet` to choose a name.

Default output is `assets/pets/NAME`, which is gitignored. Set `pet_dir` to the imported directory and restart the daemon.

Standard sheets supply idle, left/right running, waving, jumping, failure, waiting, running, and review clips. V2 sheets can also supply sixteen gaze poses. Named PNG folders and explicit custom manifests can supply fewer clips. A rest pose is required. Missing reactions use the existing rest pose; packs without left/right walking sprites stay stationary.

The importer validates referenced frames before replacing installed clip data. It preserves unrelated output files. Generated packs include a manifest and source provenance.

A custom normalized manifest declares actual clip capabilities:

```json
{
  "pet": "little-companion",
  "rest": "quiet",
  "clips": {
    "quiet": {
      "frames": ["clips/quiet/000.png"],
      "fps": 1,
      "role": "rest"
    },
    "hop": {
      "frames": ["clips/hop/000.png", "clips/hop/001.png"],
      "fps": 8,
      "role": "celebrate",
      "max_repeats": 3
    }
  }
}
```

Roles include `rest`, `work`, `attention`, `celebrate`, `error`, `greeting`, `walk-left`, `walk-right`, `gaze`, and `gesture`. Frame paths stay inside the pack. An optional `gaze` array lists sixteen static clip IDs clockwise from up. Optional `gazeAnchor` gives normalized image coordinates.

## Local protocol and checks

Clients send one JSON datagram to the daemon. Hook packets have a 2000-byte UTF-8 budget, with optional excerpts trimmed before identifiers. Missing or full daemon sockets drop hook updates without blocking the harness. Python process startup contributes to hook latency.

The daemon sends activity, playback, motion, and speech commands to the overlay. Rendering uses a background `DispatchSourceRead` and the main AppKit thread. The renderer reports geometry, interaction state, and correlated playback completion or interruption.

Speech pauses window locomotion while a stationary gesture can accompany it. Activity updates preserve the current bubble. New text replaces it. Empty text clears it.

```sh
make check
make build
```

Checks cover real UNIX sockets, native hook responses, wrapper behavior, parallel lane tracking, stale events, sprite import, animation pacing, and actual SDK serialization through a fake HTTP transport. Automated checks make no paid model requests.

Run the native desktop smoke check explicitly in a logged-in macOS session:

```sh
make desktop-check
```

It uses the built renderer, imported builtin sprites, temporary sockets, and a fake API to verify movement, hooks, long speech, expiry, and cleanup. It does not require API credentials.

For a live API and desktop check, start the daemon with your API key and run:

```sh
pet-notify speak --context "The desktop pet is running and the validation suite passed"
```

Watch for speech and its expiry, drag the pet, and exercise the parallel-session examples. Observation logs include token and prompt-cache usage.
