# Codebase guide

Simple Pet is a macOS desktop companion for coding agents. A Python daemon tracks
agent activity and generates commentary with Claude Haiku. A Swift/AppKit overlay
renders the pet. They communicate over local Unix datagram sockets.

Start with [README.md](README.md) for setup, agent hooks, configuration, and development
checks. [Makefile](Makefile) defines the check and build commands.

## Follow the work

- **Agent input:** [scripts/pet-hook](scripts/pet-hook) adapts harness events.
  [bin/pet-notify](bin/pet-notify) sends commands. [examples/](examples/) contains
  harness hook configurations and a command wrapper.
  [pet_daemon/user_text.py](pet_daemon/user_text.py) extracts bounded user prose for sentiment scoring.
- **Application flow:** [pet_daemon/daemon.py](pet_daemon/daemon.py) owns sockets,
  process startup, shutdown, and wiring. [pet_daemon/application.py](pet_daemon/application.py)
  coordinates observation ingestion, collection targets, model tasks, and delivery outcomes.
  [pet_daemon/events.py](pet_daemon/events.py) applies runtime events and reactions.
  [pet_daemon/attention.py](pet_daemon/attention.py) owns request timing, subject selection,
  fairness, and consideration history. [pet_daemon/delivery.py](pet_daemon/delivery.py)
  owns message IDs, renderer acknowledgment, and bounded retries.
- **Activity and animation:** [pet_daemon/sessions.py](pet_daemon/sessions.py)
  tracks turns, tool calls, and ownership. [pet_daemon/processes.py](pet_daemon/processes.py)
  verifies process identities and liveness. [pet_daemon/behavior.py](pet_daemon/behavior.py)
  controls movement, gestures, and naps.
  [pet_daemon/sentiment.py](pet_daemon/sentiment.py) scores user prose with VADER using the
  [bundled lexicon](pet_daemon/data/README.md). The event coordinator selects sentiment reactions and cooldowns.
- **Work and evidence:** [pet_daemon/work/](pet_daemon/work/) defines typed work,
  artifacts, relationships, evidence, and pure association and opportunity rules.
  Its views expose intent, meaningful progress, supported outcomes, and possible actions.
  [pet_daemon/sources/](pet_daemon/sources/) collects hook, Git, and GitHub observations.
  Runtime reactions remain in [pet_daemon/episodes.py](pet_daemon/episodes.py).
- **Commentary and storage:** [pet_daemon/commentary/context.py](pet_daemon/commentary/context.py)
  projects one attention selection and bounds its model context.
  [pet_daemon/commentary/cognition.py](pet_daemon/commentary/cognition.py) runs model requests,
  summaries, and repetition checks against that selection.
  [pet_daemon/commentary/model.py](pet_daemon/commentary/model.py) adapts Anthropic responses.
  [pet_daemon/storage/](pet_daemon/storage/) owns SQLite persistence, transactions, and migrations.
  Accepted speech, message outcomes, and captured-version surfacing commit together.
  The local [similarity helper](Sources/PetSpeechSimilarity/main.swift) checks repeated speech.
- **Renderer:** [Sources/PetCore/](Sources/PetCore/) defines events and socket
  handling. [Sources/PetOverlay/](Sources/PetOverlay/) contains the AppKit app and view.
- **Configuration and sprites:** [pet_daemon/config.py](pet_daemon/config.py) and
  [pet_daemon/sprites.py](pet_daemon/sprites.py) load settings and packs.
  [pet.config.example.json](pet.config.example.json) lists settings.
  [scripts/fetch-assets.sh](scripts/fetch-assets.sh) starts asset import.

## Find verification

[Tests/](Tests/) contains Python unit and runtime checks alongside Swift core,
overlay, and asset checks. Start with `test_application.py` for a complete provider-to-acceptance example,
`test_architecture.py` for dependency boundaries, `test_work.py` for the domain,
`test_work_runtime.py` for the intent-to-display path, or `test_runtime.py` for activity behavior
or the test named after the module you are changing. See the README's development
checks for automated checks, renderer builds, and desktop verification.
