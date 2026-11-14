# Command notifications

Notifications are disabled by default. Add an optional `notify` block to
`pet.config.json` to run a local command for selected agent events:

```json
{
  "notify": {
    "command": ["/absolute/path/to/notify-command"],
    "events": ["turn_done", "turn_error", "session_end", "permission_pending"],
    "filter": {
      "cwd": "/Users/me/sc/*",
      "zmx_session": "work-*",
      "labels": {"owner": "me"},
      "needs_attention": true
    },
    "debounce_seconds": 10,
    "timeout_seconds": 10,
    "permission_grace_seconds": 5,
    "max_events": 100,
    "max_attempts": 8
  }
}
```

Every field within `filter` must match. Omit fields you do not need. Cwd, zmx name,
and label values use case-sensitive shell glob patterns. `needs_attention` selects
permission prompts, turn errors, and final messages containing an error word
(`error`, `failed`, `failure`, `exception`, `unable`, or `blocked`). This is a text
heuristic. `turn_done` corresponds to a harness Stop event. Permission checks and
pending permission notifications both map to `permission_pending`, subject to the
confirmation period described below.

Use `rules` for alternatives. An event qualifies when any rule matches all its
fields. The `events` list applies to every rule. This example selects events for
one owner in any directory, plus events needing attention under a checkout root:

```json
{
  "notify": {
    "command": ["/absolute/path/to/notify-command"],
    "events": ["turn_done", "turn_error", "session_end", "permission_pending"],
    "rules": [
      {"labels": {"owner": "reviewer"}},
      {"cwd": "/Users/me/sc/*", "needs_attention": true}
    ]
  }
}
```

Rules accept the same fields as `filter`. When supplied, `rules` takes precedence
over `filter`. An empty list matches no events. Omitting `rules` uses `filter`,
whose default matches every selected event. Directory patterns are explicit and
optional, so a rule can include sessions outside a checkout root, such as
`/Users/me/.dotfiles`. Restart the daemon after changing notification settings.

Rules can also match effective workstream marks. Status and priority match exact
values. Every tag pattern must match at least one tag, using the same
case-sensitive shell globs as label values. Unmarked workstreams do not match a
rule that requires a mark. This example selects blocked workstreams, or focus
workstreams tagged for both a release and review:

```json
{
  "notify": {
    "command": ["/absolute/path/to/notify-command"],
    "rules": [
      {"marks": {"status": "blocked"}},
      {"marks": {"priority": "focus", "tags": ["release-*", "review"]}}
    ]
  }
}
```

Mark fields combine with the other fields in their rule. The daemon resolves marks
from the session's captured checkout or session thread at event acceptance.
Changing a mark affects subsequent events. Delivered `workstream_marks` contains
status, priority, and up to 32 redacted tags. Free-form mark notes, goals, and titles
stay in MCP results.

The hook input accepts optional `zmx_session`, `tab_title`, and `labels` fields.
The hook adapter also reads `ZMX_SESSION` for the zmx name. Missing metadata stays
empty unless an earlier hook supplied it for that harness and session. Labels are
a string-to-string object, such as `{"owner":"me"}`. Metadata and assistant text
are retained for at most 1,000 recent sessions. A new intent clears retained
assistant text. Only assistant progress and final messages supply that text.

The daemon executes the argv list directly and writes one JSON object to stdin.
The object contains `id`, `event`, `harness`, `session`, `zmx_session`, `cwd`,
`tab_title`, `last_assistant_message`, and `workstream_marks`. `timestamp` is the
original hook capture time when supplied, or its arrival time. `received_at` is
the daemon's arrival time. Each attempt adds `delivery_attempt_at`,
`delivery_attempt`, and `elapsed_seconds` from the original event to that attempt.
Elapsed time includes queueing, permission confirmation, and retry waits. Each
text field is capped at 2,048 UTF-8 bytes after redaction. Redaction covers common credential
assignments, authorization headers, private keys, known token prefixes, and URL
passwords. Review which sessions you include because arbitrary secrets in prose
cannot be reliably recognized.

Use a wrapper to pipe the event to curl. Save this example as an executable at
the command path above, replacing the placeholder URL when you choose a receiver:

```sh
#!/bin/sh
cat | /usr/bin/curl --fail --silent --show-error --max-time 8 \
  --config /absolute/path/to/private-curl.conf \
  --header 'Content-Type: application/json' \
  --data-binary @- https://receiver.example.invalid/events
```

Keep credentials in the wrapper's private curl config with mode `0600`. The command
inherits only a fixed system `PATH`. Put neither credentials nor event text in
command arguments or environment variables. Stdout and stderr are discarded.
Notification outcome logs contain event IDs, attempts, and status without bodies.

Permission notifications wait `permission_grace_seconds`, five seconds by default,
then confirm that the same call and turn still have a pending request in the
daemon's accepted runtime state. A tool resuming or completing that request, a new
intent, or a terminal event cancels the alert. An unrelated tool completion leaves
the pending request intact. Every retry checks again, so resolved permissions stop
retrying. Repeated observations of one request preserve its original timestamp and
confirmation deadline. A different request replaces the older pending alert.

Confirmed permission payloads include `needs_human: true` and
`permission_confirmation`. Its value is `explicit_permission_prompt` for a harness
permission prompt or `still_pending_after_grace` for a preliminary permission
check. The latter is an inference from the request remaining pending. It does not
claim that the harness reported an auto-review decision. Requests approved within
the grace period produce no permission notification. Set the grace period to zero
to check pending state immediately.

The private `pet-notify.sqlite3` beside the memory database holds redacted events
across daemon restarts and sleep. Permission alerts require a matching current
runtime request after restart. Stored permission alerts without request identity
are discarded. Other queued events retain their retry state. Delivery is at least
once, so receivers should use `id` to deduplicate. A zero exit status acknowledges
delivery.

Initial delivery attempts take precedence over retries. Other selected events are
due immediately. Failed commands and timeouts retry after one second, doubling to
a maximum wait of 15 seconds, with eight total attempts by default. The retry waits
before the eighth attempt total 60 seconds. Command execution time and queueing
add to that interval. Pending deadlines use wall time so delivery resumes after
waking. An event waiting on a retry does not delay a fresh event until that retry's
deadline.

The oldest queued events are discarded when `max_events` is exceeded, and events
are discarded after `max_attempts`. Debounce suppresses duplicate event kinds for
the same harness and session during the configured interval, including across
restarts. It drops duplicates without delaying delivery. Permissions deduplicate
by their pending request identity, so a recent completion or earlier permission
cannot suppress a distinct request. Timeout or shutdown kills the command's
process group and leaves an unacknowledged event queued. Removing `notify` stops
delivery and preserves the queue for the next time notifications are enabled.

Restart the daemon after changing notification settings or notification code.
The MCP agent and tunnel do not need a restart for notification delivery changes.
