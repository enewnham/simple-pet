# Code examples

Paths below are relative to the repository root. Read the named operation for
current details. These examples explain preferences, rather than requiring every
component to adopt the same structure.

## Vocabulary and representation

`pet_daemon/sources/github.py::_pull_request` reads GitHub's field names and maps
the state into the daemon's representation:

```python
state=PullRequestState(data["state"]),
merged_at=data.get("mergedAt"),
```

`pet_daemon/domain.py` defines the daemon's closed vocabulary.
`pet_daemon/commentary/cognition.py::Cognition.request` uses literal provider
values such as `"user"`, `"disabled"`, and `"json_schema"` in requests.
`scripts/pet-hook::EVENTS` maps harness event names at the integration boundary.

`pet_daemon/delivery.py` uses frozen records for a speech message and its outcome,
and a mutable record for the delivery attempt. `Sources/PetCore/PetEvent.swift`
uses structs for commands and playback state, with a class owning the pet model.

## Decomposition and local flow

`pet_daemon/behavior/policy.py` selects candidates from a supplied observation.
`pet_daemon/behavior/execution.py` maintains execution transitions.
`pet_daemon/behavior/controller.py` coordinates behavior and renderer commands.
This split lets a policy change stay local without moving timers or transport
details into selection rules.

`pet_daemon/work/association.py::decide_observation` decides which collected facts
to retain. `pet_daemon/work/ingest.py::WorkIngestion` applies the decision through
the repository operations its consumer needs.

`pet_daemon/behavior/policy.py::REACTIONS` is a table of regular choices.
`scripts/pet-hook::update` keeps harness-specific decisions in branches.
`pet_daemon/sessions/activity.py::mutates_files` uses a loop because parsing
depends on token order. Choose the shape that exposes the rule.

## State and boundaries

`pet_daemon/sessions/tracking.py::Sessions.activity_view` produces a snapshot of
owned session state. `pet_daemon/application.py::Application.context` enriches
snapshots with stored context through `replace`.

`pet_daemon/sources/hooks.py::normalize_hook` converts an external mapping into a
hook event. `pet_daemon/config.py::Config.load` resolves configured paths relative
to the config file. `Sources/PetCore/PetEvent.swift::PetEvent.decode` checks
incoming commands. Each gives downstream code a usable contract.

`pet_daemon/work/ingest.py::WorkIngestion.record_session` captures report context
in the transaction that records the session. `pet_daemon/commentary/presentation.py`
builds the model-facing representation explicitly from evidence views.

## Time, effects, and recovery

`pet_daemon/application.py::Application.comment` captures request revisions before
awaiting model work and checks them before delivery.
`pet_daemon/sources/repositories.py::RepositoryTracker._active` checks collection
versions and epochs before applying asynchronous observations.

`pet_daemon/delivery.py::SpeechDelivery.acknowledge` records the accepted outcome
before changing the attempt status, leaving failed persistence retryable.
`pet_daemon/work/association.py::decide_observation` preserves successful data
alongside refresh failure and success timestamps.

`Sources/PetOverlay/PetView.swift::scheduleAnimation` invalidates its owned timer
and uses a weak capture. `Sources/PetCore/SocketServer.swift` closes its descriptor
and removes the socket it owns during cancellation. `bin/pet-daemon` keeps shell
responsibility to environment setup and process launch.

## Behavioral tests

`Tests/test_delivery.py` captures packets and outcomes and advances a supplied
clock. `Tests/test_application.py` connects production components to a fake model,
a fixture observation provider, and temporary storage.
`Tests/PetCoreTests/PetCoreTests.swift::newSpeechSurvivesEarlierExpiration` captures
scheduled work so the test can invoke an earlier expiration directly.

## Guidance maintained elsewhere

`pyproject.toml` configures Ruff formatting, linting, and ty.
`Tests/test_architecture.py` enforces import boundaries. `AGENTS.md` describes
module responsibilities, storage invariants, and where to find verification.
The shared agent instructions cover conditional expressions, local variables,
unreachable guards, and comment style. Keep those rules in their existing homes.
