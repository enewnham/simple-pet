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

Drag the pet to move it. Click it for a brief remark about the work it observes.
Context and navigation buttons provide bounded follow-ups about that remark.

When work goes quiet, the pet can nap without recurring work recaps. Optional care
remarks use a small, separate request every 45–90 minutes by default and can choose
silence. Small visual care animations have their own, more frequent cadence.

## Read work through MCP

An external agent can read the pet's captured work through `bin/pet-mcp`, a stdio
MCP server. Configure your agent to launch the executable using its absolute path:

```json
{
  "mcpServers": {
    "simple-pet": {
      "command": "/absolute/path/to/simple-pet/bin/pet-mcp"
    }
  }
}
```

The server reads the configured local memory database. The daemon collects agent
hooks, reports, and repository observations. Keep the daemon running to collect
fresh activity. Each tool returns captured evidence with its observation time.

| Tool | Purpose |
| --- | --- |
| `workstreams` | Find active captured threads and their summary, branch, PR, and ticket |
| `workstream` | Read a thread's first goal, recent turns, decisions, questions, and repo state |
| `since` | Read changes after a Unix timestamp, grouped by thread |
| `sessions` | Read fresh harness sessions and their runtime state |
| `attention` | Find pending permissions and follow-up candidates |
| `decisions` | Find decision and correction excerpts with their sources |
| `mark_workstream` | Record caller judgments, titles, goals, tags, and snoozes |
| `unmark_workstream` | Clear selected marks while preserving history |
| `workstream_mark_history` | Read attributed mark and clear events |
| `context_search` | Search captured work and reports |
| `read_passage` | Read a source with surrounding text and provenance |

Use `--config /path/to/pet.config.json` to select another configuration, or
`--memory-file /path/to/pet-memory.sqlite3 --pet manifest-name` to select stored
records directly. Search and context results are bounded. Use `read_passage` to
continue reading a source. `since` does not retain deleted sources or
superseded repository observations.

Start with `workstreams`, then pass a returned ID to `workstream`. Results preserve
reported versus observed evidence and source timestamps. Caller marks have separate
attribution and append-only history. Summaries use captured
text. Decision and question excerpts are labeled candidates. `sessions` requires
a fresh daemon snapshot and verified harness processes. Paths have no assumed root.

See [MCP tools and evidence](docs/mcp.md) for grouping, freshness, limits, attention
rules, and compatibility tools. `current_work_context` and `relevant_changes` remain
available for clients using flat source lists.

Restart the daemon with `make restart-agent` after notification configuration or
code changes. Restart the HTTP MCP server with `make restart-mcp-agent` after MCP
code changes. For stdio MCP, reconnect the client to start a fresh server process.

Your external agent owns conversation, investigation, and execution. Connect that
agent directly to Linear, meetings, or other services it needs.

### Remote access through ngrok

Stdio is the default transport. Start Streamable HTTP explicitly:

```sh
bin/pet-mcp --http --host 127.0.0.1 --port 18765
```

The endpoint is `http://127.0.0.1:18765/mcp`. The server accepts only loopback bind
addresses and exposes the same evidence and workstream mark tools. Every MCP request requires
an `Authorization: Bearer <token>` header. Missing or invalid credentials return
401. This uses a shared bearer credential, with no interactive OAuth login.

HTTP startup creates `state/pet-mcp.token` with mode `600` if it is missing.
Existing token files must belong to your user and have exactly those permissions.
Use `--token-file /private/path/token` to select another file. The server never
prints the token and disables HTTP access logs. Keep the token in your remote
client's protected credential field, separate from its endpoint URL. To transfer
it through the macOS clipboard without printing it:

```sh
pbcopy < state/pet-mcp.token
```

To run the HTTP server at login and restart it after failures:

```sh
make install-mcp-agent
```

The user LaunchAgent is `dev.simple-pet.mcp`. It defaults to `127.0.0.1:18765`.
Its logs are `~/Library/Logs/Simple Pet/mcp-stdout.log` and `mcp-stderr.log`.
Use `make restart-mcp-agent`, `make status-mcp-agent`, and
`make uninstall-mcp-agent` to manage it. Stop a foreground HTTP server before
installing the agent. To rotate the credential, remove the token file and restart
the service, then update your remote client's credential.

Install and authenticate the [ngrok agent](https://ngrok.com/docs/gateway/endpoints/agent-cli-quickstart),
then install the persistent tunnel with your ngrok domain:

```sh
make install-tunnel-agent ARGS='--domain your-domain.ngrok.app'
```

The user LaunchAgent is `dev.simple-pet.tunnel`. It forwards the configured HTTPS
endpoint to `http://127.0.0.1:18765`. The domain is stored in its plist and can be
changed by reinstalling with another `--domain`. The MCP agent runs separately.
Use `make restart-tunnel-agent`, `make status-tunnel-agent`, and
`make uninstall-tunnel-agent` to manage the tunnel. Uninstalling retains credentials.

Installation creates `state/pet-mcp-tunnel.secret` and renders
`state/pet-mcp-ngrok.yml` from [the policy template](examples/pet-mcp-ngrok.yml).
Both private files have mode `600` and stay outside version control. The secret
survives reinstall and restart. Restarting renders the template again. To rotate
the tunnel secret, remove its file and restart, then update the remote client.

The traffic policy denies requests unless `x-pet-tunnel-key` contains exactly the
shared secret. It permits only GET, POST, and DELETE on `/mcp` and rewrites the
upstream Host to `127.0.0.1:18765`. The local MCP server also requires its separate
bearer token. These rules use ngrok's
[header variables](https://ngrok.com/docs/gateway/traffic-policy/variables/req),
[request rules](https://ngrok.com/docs/gateway/traffic-policy/examples/block-unwanted-requests),
and [Host header override](https://ngrok.com/docs/gateway/traffic-policy/actions/add-headers).

The endpoint can also carry other local MCP servers on their own paths. Each server
runs its own ngrok agent with an [internal endpoint](https://ngrok.com/docs/universal-gateway/internal-endpoints/)
such as `https://agent-rc.internal`. List the routes in `state/pet-tunnel-routes.json`, in
the shape of [the example](examples/pet-tunnel-routes.json):

```json
[{"path": "/agent-rc/mcp", "forward": "https://agent-rc.internal"}]
```

Each route forwards GET, POST, and DELETE on exactly that path with
[forward-internal](https://ngrok.com/docs/traffic-policy/actions/forward-internal), after the
`x-pet-tunnel-key` check, so routed requests need the same header. The internal endpoint's
own policy sets its Host header and path, and its server checks its own bearer token. Paths
must differ from `/mcp`, and targets must be `https://<name>.internal`. Restart the tunnel
after changing routes.

Configure your remote MCP client for Streamable HTTP at
`https://your-domain.ngrok.app/mcp` with these protected
credentials:

| Header | Private source |
| --- | --- |
| `Authorization` | `Bearer ` followed by the value in `state/pet-mcp.token` |
| `x-pet-tunnel-key` | Value in `state/pet-mcp-tunnel.secret` |

To copy the tunnel secret without printing it:

```sh
pbcopy < state/pet-mcp-tunnel.secret
```

The public endpoint returns 403 without the tunnel header and 401 with that header
but no bearer token. Both credentials permit MCP initialization and `tools/list`.
Clients should omit the browser `Origin` header because the MCP server accepts
loopback origins only.

Tunnel lifecycle logs are `~/Library/Logs/Simple Pet/tunnel-stdout.log` and
`tunnel-stderr.log`. The launcher suppresses ngrok's raw output so policy errors
cannot print the embedded secret. Ngrok request inspection is disabled. Keep the
account's [Traffic Inspector Full Capture](https://ngrok.com/docs/obs/traffic-inspection/http-requests)
disabled and exclude headers and bodies from log exports. Full Capture stores
unredacted credentials and work evidence.

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

## Session naming

Session naming is optional and runs locally. Add a `naming` command callback to
receive short names derived from workstream titles, goals, or first prompts. See
[session naming](docs/session-naming.md) for configuration and the zmx title example.

## Command notifications

Notifications are disabled by default. Add `notify` to `pet.config.json` to run a
local command for selected agent events:

```json
{
  "notify": {
    "command": ["/absolute/path/to/notify-command"],
    "events": ["turn_done", "session_end", "permission_pending"],
    "filter": {"cwd": "/Users/me/sc/*", "needs_attention": true}
  }
}
```

The command receives a JSON event on stdin with session metadata and the last
assistant message, capped at about 2 KB with common secrets redacted. For example,
save this curl wrapper as an executable at the command path above:

```sh
#!/bin/sh
cat | /usr/bin/curl --fail --silent --show-error --max-time 8 \
  --config /absolute/path/to/private-curl.conf \
  --header 'Content-Type: application/json' \
  --data-binary @- https://receiver.example.invalid/events
```

Choose your receiver URL and keep credentials in the private curl config with
mode `0600`. Keep secrets out of command arguments and environment variables.
Use `notify.rules` for OR between filters. Each rule requires all its fields to
match. See [notification configuration and delivery](docs/notifications.md) for filters,
the event contract, retry settings, and delivery guarantees.

## License and attribution

Simple Pet is a fork and rebuild of [OpenPets](https://github.com/OpenPetsHQ/openpets). Credit goes to the OpenPets authors for the original project and artwork.

Simple Pet is licensed under the [MIT license](LICENSE), retaining OpenPets’ copyright notice.
