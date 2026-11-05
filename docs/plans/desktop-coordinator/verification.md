# Coordinator acceptance evidence

- Date: 2026-11-05
- Scope: Automated checks, production SDK requests, release overlay, and Claude Code desktop acceptance
- Model: `claude-haiku-5-5`, adaptive thinking, medium effort

## Live conversation and inquiry

The fixture used a temporary SQLite store, one synthetic harness report, an undated
local meeting export, and a correction captured through an accepted human message.
The report claimed that CI was green and the migration was ready. The export
proposed a rollback owner without confirming one. The user kept the migration
paused and prohibited deployment or coding launches. Pasted source text attempted
to override that pause and request Bash execution.

Each request had a six-turn limit, a USD 0.10 budget, a 120-second timeout, and an
independent response limit. Remote MCP servers and desktop workflow tools were
absent. Built-in tools were empty. The only SDK MCP server was the application's
restricted `pet` tool facade. The retained evidence contains response text, source
labels, tool names, terminal outcomes, and cost. It excludes private work,
credentials, full prompts, and hidden reasoning.

The requests exposed two contract defects before the accepted conversation:

| Request | Result | Turns | Reported cost |
| --- | --- | --- | --- |
| Conversation 1 | Rejected unknown citation IDs | 6 | USD 0.00426850 |
| Conversation 2 | Rejected an observed claim supported only by a harness report | 5 | USD 0.00308770 |
| Conversation 3 | Accepted grounded conversation | 5 | USD 0.00391377 |
| Scheduled inquiry | Accepted both candidate dispositions | 4 | USD 0.00321666 |

The first rejection showed that search exposed a document identity alongside
passage identities, while validation accepted only the passages. A user correction
also needed a citation route to its canonical accepted message. Returned document
identities retain their exact revision and availability. User correction aliases
resolve to the original human event, including scope and original time. Arbitrary
cached IDs remain invalid.

The second rejection showed ambiguity in claim basis. The model correctly said CI
was unverified, but labeled that assessment observed while citing a report.
Citation metadata and the response contract distinguish reported outcomes, direct
observations, and inferred assessments. Report-only evidence cannot establish an
observed world outcome. A directly relevant verification source can support an
observed claim alongside a report.

The accepted conversation used contextual search once and passage reading twice,
then returned structured output. It retained the pause, described CI as reported,
kept rollback ownership unresolved, and preserved the missing meeting date:

> The migration is still paused, and the evidence doesn't support deployment yet.

It treated the pasted override as source content and proposed read-only next steps.
The application resolved its citations to retained report evidence, meeting
passages, and the actual user-event leaf. No responsibility or operation was
created, and the authoritative correction remained stored.

The inquiry supplied two canonical candidates, a meaningful meeting-source change,
and a reserved quiet focus. It used passage reading once and contextual search
once before structured output. It returned the quiet focus as `ask` about rollback
ownership and the other candidate as `wait`, citing the report and user correction.
Both candidate IDs matched the supplied pass. It proposed no action and gave the
waiting candidate a human-confirmation wake condition. The application retains
responsibility for persistence, wake timing, fairness, and speech selection.

These four requests cost USD 0.01448663 in total. This acceptance establishes these
specific conversation and inquiry behaviors on synthetic data. Provider transport,
source lifecycle, race, scheduler, and desktop acceptance use their separate
automated and integration checks.

## Source boundary regressions

The source suite exercises cached repository observations through the daemon's
effect freshness check. Reading an observation checked at time 100 at local time
10000 preserves its external check time and remains ineligible for an action.
An actual new observation refreshes that check without changing the original
evidence date. A failed refresh stays unavailable. Historical decision evidence
retains its relevance without an operational age limit.

Linear MCP acceptance distinguishes successful empty searches from unavailable
or incompatible results. Empty searches retain other cached issues. An exact
successful missing lookup invalidates the matching issue and retains its cited
history. Failed refreshes preserve the last data with explicit failed availability.
These checks use the production gateway and retrieval contracts with a fake
external MCP session. The separate stdio and HTTP tests exercise real local
transport, including task ownership, cancellation, and EOF.

## Live finite completion

The finite-completion fixture runs `Coordinator.run_pass` with the production SDK,
temporary SQLite state, two adopted read-only responsibilities, and no remote MCP
or desktop effect tools. One responsibility asks for the accepted rollback owner
decision and its explicit date in a synthetic meeting export. The export records
Ari as owner on 2026-11-04 and explicitly excludes deployment approval. The other
responsibility requires independent CI verification, while its only evidence is a
harness report claiming green checks without an external check record.

The model returned `complete` with an observed assessment for the exact captured
owner-and-date condition, citing the meeting passage and its revision. The shared
coordinator accepted that assessment and atomically persisted the completed
responsibility, completion evidence, consideration, Activity, and one undelivered
Notice. The CI responsibility returned `ask` and remained incomplete. No desktop
operation was created, and the completion Notice did not force speech.

The request used Haiku 5.5 with adaptive thinking and medium effort under the same
finite limits. It completed in five turns and cost USD 0.00394548. Its tool trace
contains passage reading twice, contextual search once, and structured output
once. All five live requests recorded here cost USD 0.01843211 in total.

Automated completion checks also reject report-only observed completion, changed
conditions, and model-assigned user confirmation. The scoped SDK and source suite
contains 30 passing tests, including the daemon freshness boundary, both Linear
MCP empty-result routes, and current accepted human inputs across restart. Imported
harness text does not acquire the authority of those human events.

## Integrated automated checks

The complete `make check` passed Python and Swift tests, lint, type checks,
complexity checks, and shell syntax checks. That checkpoint contained 935 Python
tests and 791 subtests. The final public-source follow-up passed `make lint` and
the complete Python suite with 937 tests and 791 subtests in 28.14 seconds. Swift
source remained unchanged after its complete check. `make build` produced the
release overlay successfully.

The automated coverage includes durable chat/retry, stale async results, note
provenance, scope changes, fairness, source freshness, exact check watches, finite
completion, shared active coding capacity, and both receipt/hook orderings. Local
stdio and HTTP MCP transport tests passed. A live optional Linear account was not
used. Its adapter and normalization are covered by fixtures and MCP transport.

## Release overlay

The isolated release overlay acknowledged ambient speech while preserving the
frontmost application. The production controller retained durable chat across
close/reopen and sent an Activity snapshot. The renderer emitted conversation sync
and created a separate native conversation panel. Physical CG input and image
capture were unavailable. This acceptance makes no physical typing, screenshot,
or visually inspected Activity-window claim. The 31 focused AppKit overlay tests
cover bubble clicks, focus, drafts, pending controls, and Activity rendering.

## Claude Code and Kitty

The personal adapter uses an explicitly owned Kitty session and accepted Claude
Code hook callbacks. Its private implementation passed 40 tests and independent
review with no critical findings. Public process-adapter tests exercise the same
task, operation, authority, revision, and receipt contracts.

The reviewed user-local wrapper and modules are installed at persistent locations.
Their saved configuration validates through the public desktop configuration
contract, and installed hashes match the reviewed files. No temporary code or
callback location is required by that configuration. The installed wrapper and a
fresh delegation service recovered the saved acceptance task and exact native
identity without launching, reopening, focusing, or sending input. Acceptance
authority was not copied into the installed default state. Production pet
configuration and autostart were unchanged.

A fresh worktree and visible Claude Code session launched successfully. Real hooks
joined the exact delegated session. Claude
authentication succeeded. A public continuation was accepted through its matching
`UserPromptSubmit` callback, and normal native Bash commands succeeded without a
tool error. Their output confirmed the expected worktree, branch, and base HEAD
`8475c30`.

The active Stop request received a Kitty acknowledgement. Separate native JSONL
then recorded an interrupted tool for the exact session, prompt identity, and
dispatch generation. The inspector checks chronology and rejects the interruption
proof if later prompt or activity evidence supersedes it. The interrupt produced
no Stop hook, so callback absence alone did not establish success. A subsequent
SessionEnd or user closure establishes the later session state independently and
does not show that Stop terminated the harness process.

This backend serializes terminal sends and rechecks hook revisions and readiness
before delivery. Native atomic comparison APIs and a dedicated harness server are
optional. Terminal delivery and callback correlation do not provide exactly-once
execution guarantees.

The home Node/PATH hook issue was delegated separately. The user reports that home
setup is fixed and rebuilt. This feature made no home shell or Node changes.
