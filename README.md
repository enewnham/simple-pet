# Simple Pet

A macOS desktop pet that speaks about your coding agents’ work.

![Simple Pet demo](docs/media/simple-pet.webp)

## Quick start

Requires macOS 12+, Swift 6.0+, Git, [uv](https://docs.astral.sh/uv/), an Anthropic API key, and a pet sprite pack.

Find pets at [OpenPets packs](https://openpets.dev/packs) or [OpenPet Hatchery](https://github.com/chenmi319/openpet-hatchery). Download a V2 pet ZIP, then run these commands from the repository root:

```sh
make setup
make build
scripts/fetch-assets.sh --zip ~/Downloads/pet.zip --pet my-pet
cp pet.config.example.json pet.config.json
export ANTHROPIC_API_KEY='your-api-key'
bin/pet-daemon
```

### Run in the background

The pet can run as a daemon and start automatically at login.
Save your API key in `.env` at the repository root:

```sh
ANTHROPIC_API_KEY='your-api-key'
```

Install:

```sh
make install-agent
```

Use `make restart-agent` after changing configuration and `make uninstall-agent`
to stop the pet and remove it from login startup.

Edit `pet.config.json` to customize the pet's voice and
settings, then restart. If you imported a pack with a different name, set `pet_dir`
to its directory under `assets/pets/`.

Drag the pet to move it. Click it to start a conversation.

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

For commands outside a coding tool, use the wrapper:

```sh
examples/pet-run pytest
```

## License and attribution

Simple Pet is a fork and rebuild of [OpenPets](https://github.com/OpenPetsHQ/openpets). Credit goes to the OpenPets authors for the original project and artwork.

Simple Pet is licensed under the [MIT license](LICENSE), retaining OpenPets’ copyright notice.
