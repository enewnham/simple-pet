# Simple Pet

A macOS desktop companion for your ongoing work.

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

## Conversation and work context

Click the pet to ask about your work, correct its understanding, or adopt a bounded
responsibility. Conversation history, corrections, responsibilities, and task
outcomes survive daemon restarts. Activity shows investigations and delegated work.
Background inquiries retain recent replies and constraints from your pet conversation.
Pausing coordination and stopping a desktop task are separate controls.

The coordinator uses Claude Haiku 5.5 with adaptive thinking and medium effort.
`make setup` installs the Python Claude Agent SDK and its bundled CLI. The
coordinator has separate model, timeout, turn, response, and cost settings under
`coordinator` in [pet.config.example.json](pet.config.example.json). Ambient speech
retains its own model settings. Desktop implementation uses a configured external
adapter, described in [Desktop adapters](docs/desktop-adapter.md).

When work goes quiet, the pet can nap without recurring work recaps. Optional care
remarks use a small, separate request every 45–90 minutes by default and can choose
silence. Small visual care animations have their own, more frequent cadence.

The pet starts with stored work reports. To include existing harness history and
local meeting exports, merge this into your configuration:

```json
{
  "coordinator": {
    "history": [
      {
        "name": "codex",
        "path": "~/.codex/sessions",
        "format": "codex"
      }
    ],
    "meetings": {
      "folders": ["~/Documents/meeting-exports"]
    }
  }
}
```

History reads are bounded by configured file, byte, and record limits. Supported
formats are Claude and Codex JSONL session records and JSONL message exports with
`role` and `content` fields. Historical sessions supply context without granting
control of those sessions. Meeting exports use Markdown or plain text. Edits and
removals appear after source refresh, including reconciliation after wake. An
undated export retains unknown chronology.

Optional Linear access uses a read-only ticket interface. Enable
`coordinator.linear`, select team or assignee IDs, or configure followed and linked
issue IDs. Set the API key in the environment variable named by
`authorization_env`, which defaults to `LINEAR_API_KEY`. Exact issue lookups and
scoped searches retain Linear identities, versions, links, and availability.
Keep credentials in your process environment or the background agent's `.env`.

Selected MCP operations can supplement contextual search. An HTTP example is:

```json
{
  "coordinator": {
    "mcp": [
      {
        "name": "work",
        "type": "http",
        "url": "https://example.com/mcp",
        "authorization_env": "WORK_MCP_AUTHORIZATION",
        "read_only": true,
        "tools": ["search"],
        "search_tool": "search",
        "query_parameter": "query"
      }
    ]
  }
}
```

Replace the endpoint and tool names with those supplied by your server.
`authorization_env` contains the complete authorization header value. A stdio
connection uses `type: "stdio"`, `command`, `args`, and an explicit `environment`
list of credential variable names to forward. Select exact remote tool names
without wildcards. The server must declare each selected tool read-only and must
not declare destructive effects. Missing tools or incompatible declarations remain
visible as source failures. Unselected tools are excluded from the pet's tool
facade, and project settings, hooks, and coding tools are not inherited.

`search_tool`, `query_parameter`, and optional `search_arguments` connect a selected
operation to contextual search. `bootstrap_query` enables a configured initial
query. Precise selected operations remain available without a collector. For a
Linear MCP integration, set `provider: "linear"`, select its lookup operation with
`lookup_tool`, and set `identifier_parameter` to its issue-ID argument. Complete
versioned Linear issue results share identities and revisions with the API adapter.
Partial results remain cited source evidence with their limits visible.

The pet reports unavailable sources and uncertainty. Source text cannot adopt work
or override a user correction. Clearing pet memory also discards cached SDK
conversation context. Configured external history remains readable until its
source configuration is removed.

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
