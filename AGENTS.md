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
- **Daemon coordination:** [pet_daemon/daemon.py](pet_daemon/daemon.py) receives
  events and manages process lifecycle. [pet_daemon/events.py](pet_daemon/events.py)
  coordinates session updates, memory, and reactions.
  [pet_daemon/commentary.py](pet_daemon/commentary.py) schedules observations, rejects
  stale responses, and submits speech. [pet_daemon/delivery.py](pet_daemon/delivery.py)
  owns speech acceptance, delivery IDs, and bounded retries. [pet_daemon/attention.py](pet_daemon/attention.py)
  selects commentary subjects and cooldowns.
- **Activity and animation:** [pet_daemon/sessions.py](pet_daemon/sessions.py)
  tracks turns, tool calls, and ownership. [pet_daemon/processes.py](pet_daemon/processes.py)
  verifies process identities and liveness. [pet_daemon/behavior.py](pet_daemon/behavior.py)
  controls movement, gestures, and naps.
  [pet_daemon/sentiment.py](pet_daemon/sentiment.py) scores user prose with VADER using the
  [bundled lexicon](pet_daemon/data/README.md). The event coordinator selects sentiment reactions and cooldowns.
- **Work and evidence:** [pet_daemon/work/](pet_daemon/work/) stores intent, sessions,
  artifacts, relationships, and attributed evidence. Its opportunity derivation and
  work views feed attention. [pet_daemon/repositories/](pet_daemon/repositories/) supplies
  independent Git and GitHub observations. Runtime reactions remain in
  [pet_daemon/episodes.py](pet_daemon/episodes.py).
- **Commentary and memory:** [pet_daemon/cognition.py](pet_daemon/cognition.py)
  sends model requests and interprets responses.
  [pet_daemon/context.py](pet_daemon/context.py) selects and bounds model context.
  [pet_daemon/summarization.py](pet_daemon/summarization.py) builds session summaries
  from requests and agent reports.
  [pet_daemon/repetition.py](pet_daemon/repetition.py) checks recent speech through the local
  [similarity helper](Sources/PetSpeechSimilarity/main.swift).
  [pet_daemon/memory.py](pet_daemon/memory.py) stores bounded SQLite history.
- **Renderer:** [Sources/PetCore/](Sources/PetCore/) defines events and socket
  handling. [Sources/PetOverlay/](Sources/PetOverlay/) contains the AppKit app and view.
- **Configuration and sprites:** [pet_daemon/config.py](pet_daemon/config.py) and
  [pet_daemon/sprites.py](pet_daemon/sprites.py) load settings and packs.
  [pet.config.example.json](pet.config.example.json) lists settings.
  [scripts/fetch-assets.sh](scripts/fetch-assets.sh) starts asset import.

## Find verification

[Tests/](Tests/) contains Python unit and runtime checks alongside Swift core,
overlay, and asset checks. Start with `test_work.py` for the domain,
`test_work_runtime.py` for the intent-to-display path, or `test_runtime.py` for activity behavior
or the test named after the module you are changing. See the README's development
checks for automated checks, renderer builds, and desktop verification.
