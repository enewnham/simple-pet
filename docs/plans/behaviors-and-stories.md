# Behaviors and stories

## Context

Simple Pet is an ambient persona that shares the developer's workspace. Its pack
supplies a body and its persona configuration supplies its identity, voice, and
quirks. It can be dry, nosy, or a little snarky, and it wants to keep the developer
company. Agent work gives it something to care about. Long reports belong in
conversation after a click. Automatic remarks need a small reason to speak.

## Decision

The daemon keeps a mood, chooses one occasion for an automatic remark, and
remembers occasions whose lines the renderer acknowledged. It uses its existing
session events, reports, repository observations, and caller notes. Summarizing work
and speaking are separate model calls.

### Mood and idle

`behavior/mood.py` persists energy, curiosity, appetite for company, and agitation.
Wall time advances decay and ages stored observations across sleep and restart.
Timing that only schedules a live callback can use monotonic time. Recent work,
genuine novelty, and human engagement are separate signals. A quiet tool can make
the pet bored while a click still resets its appetite for company. VADER contributes
a small agitation cue from prose associated with a known human prompt.

After a fresh reaction, the pet may offer company once. As the quiet stretches
on, movement and rest take over. An acknowledged offer starts the unanswered-bid
timer. A response or new work clears it. System sleep clears the timer so sleep
doesn't turn into a complaint on wake. A long absence followed by a click or a new
available presence observation can prompt a greeting. Background agent output
never counts as developer engagement.

### Occasions and work

An occasion carries a fingerprint, a short fact or the source for deriving it, a
speech situation, and a freshness deadline. Use the existing hooks and attention
triggers to decide why the pet speaks. A table sets the feeling, model effort,
and plain-language instruction for the situation. The model writes the line in the
configured voice and can choose the joke, rhythm, or degree of snark.

| Situation | Input and expiry |
| --- | --- |
| Trouble and recovery | A small timer keeps the latest repeated failure. Success on the same operation can close it in a later turn. Cancellation and age end the opportunity. |
| Long wait | Use the active tool's start time or the existing request for input. The occasion stops when the tool or wait changes. |
| Return | A direct human signal after a quiet period supplies a short-lived greeting. |
| Unanswered company | A timer starts after the renderer acknowledges an offer. It clears on engagement, new work, or system sleep. |
| Handoff or new contributor | A recent typed work note supplies a reaction while the associated work is visible. |
| Shared result | Equivalent reports from separate sessions on the same PR or linked parent work share a fingerprint. |
| Changed direction | Explicit language in a new request can give the pet a fresh occasion. Treat the wording check as a best-effort cue. |

The fingerprint comes from the source work, tool call, or presence event before
model generation. Repeated equivalent reports about the same result share it.
The acknowledged line and fingerprint commit together. The fingerprint remains
in the `said` history beyond the occasion's freshness window. Failed delivery and
silent model results add no spoken record. The wording filter remains a last check
against recent speech and retains its three-line and identical-content-token
limits.

### From event to line

1. Accept and record a hook or observation. Session tracking already knows the work
   state, tool calls, waits, and outcomes.
2. Apply mood cues and update the failure or unanswered-offer timer as needed.
3. Let `attention.py` enforce timing, novelty, and fairness. `occasions.py` selects
   one reason to speak. Work descriptions turn selected evidence into short facts.
4. If the fact requires interpretation, the story model maintains the supported
   work summary. The speech model sees one fact with its supporting sources,
   the situation's instruction, and the pet's current feeling. The full work story
   remains available for click-through conversation.
5. Recheck that the selected work or wait is current before delivery. Renderer
   acknowledgment records the spoken fingerprint.

### Existing animation capabilities

The overlay plays whatever capabilities the selected pack provides. Missing
capabilities put the expression in voice, timing, or silence.

| Behavior | Existing expression |
| --- | --- |
| Energy and focus | Work bursts, movement, and rest |
| Curiosity | Roam or gaze where supported |
| Company and return | Greet or wave, optionally with one line |
| Trouble or interruption | Distress or consternation, then a pause |
| Recovery and shared success | Recover, jump, or celebrate |
| Long quiet or an unanswered offer | Less movement, gestures, and rest or sleep |
| Handoff and changed direction | Work or roam if appropriate. Voice carries the relationship or change. |

Preserve pointer attention where available, occasional greetings, short work
bursts, recovery gestures, periodic rest, and dry remarks. The cat pack is an
example with no gaze capability. Pack-specific play, mischief animations, and new
gaze poses need new art and are deferred.

### Outside knowledge

The pet needs to know who did the work, who they were acting for, what they are
waiting for, what was passed to someone else, and whether a human was present.
`mark_workstream` accepts typed notes for assignments, launches, waits, and
handoffs. Each writer maintains its own note list, and readers see all current
notes with their provenance. The same tool handles shared status and goal marks. Each actor
has a name and optional description. The pet can talk about agents as other
participants and addresses the developer as “you.”

`record_presence` reports available or away. An integration can include the
session and turn of a human prompt, or link an observation to a renderer click.
The renderer owns responses to clicks. The HTTP read credential grants reads,
and the optional write credential grants the same tools plus writes. Local stdio
already runs under the user's account. Notes remain attributed caller records,
separate from captured evidence. [MCP tools and evidence](../mcp.md)

## Consequences

The behavior and spoken-history schema use the repository's local database
recreation policy. Workstream marks live in a separate file. Equivalent source
facts share an occasion when the code can establish their identity. Rephrased
reports can still need the wording filter. Pack capabilities determine how much
of the personality can be expressed without a line.

## Open questions

- Which kinds of quiet invitations feel welcome after using the pet for a day?
- Which situations reliably produce a personal remark with the configured model?
  Use captured examples and a desktop trial to answer this.
- Which optional rituals work through timing, voice, and the existing pack
  capabilities? Pack-specific play and mischief can be revisited when art is
  available.
