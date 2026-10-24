# Codebase guide

Simple Pet is a macOS desktop companion for coding agents. A Python daemon tracks
agent activity and generates commentary with Claude Haiku. A Swift/AppKit overlay
renders the pet. They communicate over local Unix datagram sockets.

Start with [README.md](README.md) for setup, agent hooks, configuration, and development
checks. [Makefile](Makefile) defines the check and build commands.

Prefer `make` targets for local setup, builds, formatting, linting, and tests.
Run focused tests when they cover the change.

```sh
# All tests
make test
make test-python
make test-swift

# Run specific tests
make test-python ARGS='Tests/test_notify.py'
make test-swift ARGS='--filter PetCoreTests'
```

## Code style

When writing, refactoring, or reviewing code, load the
[code-style skill](.agents/skills/code-style/SKILL.md) for repository conventions
that require judgment beyond formatting, linting, and type checks.

## Follow the work

- **Agent input:** [scripts/pet-hook](scripts/pet-hook) adapts coding tool events.
  [bin/pet-notify](bin/pet-notify) sends commands. [examples/](examples/) contains
  coding tool hook configurations and a command wrapper.
  [pet_daemon/user_text.py](pet_daemon/user_text.py) extracts bounded user prose for sentiment scoring.
- **Application flow:** [pet_daemon/daemon.py](pet_daemon/daemon.py) owns sockets,
  process startup, shutdown, and wiring. [pet_daemon/application.py](pet_daemon/application.py)
  coordinates observation ingestion, collection targets, model tasks, and delivery outcomes.
  [pet_daemon/sessions/coordinator.py](pet_daemon/sessions/coordinator.py) coordinates accepted hooks,
  activity reactions, and commentary triggers.
  [pet_daemon/attention.py](pet_daemon/attention.py) owns request timing, subject selection,
  fairness, and consideration history. [pet_daemon/delivery.py](pet_daemon/delivery.py)
  owns message IDs, renderer acknowledgment, and bounded retries.
- **Activity and animation:** [pet_daemon/sessions/tracking.py](pet_daemon/sessions/tracking.py)
  tracks turns, tool calls, and runtime presence. [pet_daemon/sessions/processes.py](pet_daemon/sessions/processes.py)
  verifies process identities and liveness. [pet_daemon/behavior/controller.py](pet_daemon/behavior/controller.py)
  controls movement, gestures, and naps.
  [pet_daemon/sentiment.py](pet_daemon/sentiment.py) scores user prose with VADER using the
  [bundled lexicon](pet_daemon/data/README.md). The activity coordinator selects sentiment reactions and cooldowns.
- **Work and evidence:** [pet_daemon/work/](pet_daemon/work/) defines typed agent sessions,
  report contexts, checkout facts, and explicit PR, stack, and ticket links.
  Runtime session transitions supply accepted facts to [pet_daemon/work/ingest.py](pet_daemon/work/ingest.py).
  [pet_daemon/work/shared_work.py](pet_daemon/work/shared_work.py) joins stored sessions, reports,
  and repository observations into shared work stories with distinct session contributions.
  Report references can resolve later without
  using a checkout's later state to reinterpret the report.
  [pet_daemon/sources/](pet_daemon/sources/) collects hook, Git, and GitHub observations
  without assigning work to sessions or creating PR tasks.
  [pet_daemon/sessions/reactions.py](pet_daemon/sessions/reactions.py) tracks reactions to accepted session transitions.
- **Commentary and storage:** [pet_daemon/commentary/request.py](pet_daemon/commentary/request.py)
  builds bounded commentary prompts and captures their supporting facts.
  [pet_daemon/commentary/cognition.py](pet_daemon/commentary/cognition.py) runs model requests,
  summaries, and repetition checks against that selection.
  [pet_daemon/commentary/model.py](pet_daemon/commentary/model.py) adapts Anthropic responses.
  [pet_daemon/storage/](pet_daemon/storage/) owns SQLite persistence, transactions, and migrations.
  Session metadata, report context, and request history commit together.
  Reports retain their observed references; current checkout state remains separate. Accepted speech,
  message outcomes, and captured-subject consideration commit together.
  Schema changes always drop and recreate local storage. Never add data-preserving
  migrations. Newer schemas are rejected before writes.
  [Speech repetition checks](pet_daemon/commentary/repetition.py) use Apple's sentence embeddings through PyObjC.
- **Renderer:** [Sources/PetCore/](Sources/PetCore/) defines events and socket
  handling. [Sources/PetOverlay/](Sources/PetOverlay/) contains the AppKit app and view.
- **Configuration and sprites:** [pet_daemon/config.py](pet_daemon/config.py) and
  [pet_daemon/behavior/sprites.py](pet_daemon/behavior/sprites.py) load settings and packs.
  [pet.config.example.json](pet.config.example.json) lists settings.
  [scripts/fetch-assets.sh](scripts/fetch-assets.sh) starts asset import.

## Find verification

[Tests/](Tests/) contains Python unit and runtime checks alongside Swift core,
overlay, and asset checks. Start with `test_application.py` for a complete provider-to-acceptance example,
`test_architecture.py` for dependency boundaries, `test_work.py` for report attribution and observed associations,
`test_connected_work.py` for parent and subagent work across checkout layouts,
`test_work_runtime.py` for the intent-to-display path, or `test_runtime.py` for activity behavior
or the test named after the module you are changing. See the README's development
checks for automated checks, renderer builds, and desktop verification.
