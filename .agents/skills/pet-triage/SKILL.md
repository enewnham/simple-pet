---
name: pet-triage
description: Investigate a recent Simple Pet behavior such as speech, silence, reactions, naps, movement, or timing using local evidence and replay experiments.
---

# Pet triage

Investigate the behavior the user names. Treat the logs, SQLite store, active configuration, and daemon code as evidence. The visible symptom may come from a hook, timer, model request, delivery acknowledgment, or renderer state, so follow the event path that fits the case.

## Start cheaply

- Resume an existing case from `case.json`, saved results, and its investigation script. Presentation changes render saved results. Extend evidence or rerun an experiment when the question or inputs change.
- Load live logs once, preserve the relevant records, and print a compact event index. Inspect selected request fields or the matching session in a snapshot. Keep full payloads in artifacts.
- Choose the smallest production entrypoint that answers the question. Check live model availability before preparing model experiments, and compare the running daemon's recorded configuration with the working tree before source-level replay.
- Use the [investigation recipes](references/investigation.md) for cached capture, compact inspection, and choosing a replay boundary. Keep collection, experiments, and report rendering separately callable in case scripts.

## Establish the case

- Record the user's question, approximate time, expected and observed behavior, checkout revision, config path and relevant values, pet pack, and model. Distinguish facts captured during the incident from details reconstructed later.
- Read the configured JSONL log and its rotations with `pet_daemon.triage.load_logs`. Keep each record's source path and line number in citations. Use `select_logs` for a bounded time window and exact fields. Inspect malformed line diagnostics and rotation gaps before drawing conclusions from absence.
- Correlate events by `run_id`, `observation_id`, session, and structured packet fields as the case permits. `trace_delivery` follows a speech message ID through a delivery record and renderer command. A UUID inside user or agent prose is content, not an event link.
- Query the local SQLite file directly or reuse daemon storage methods for relevant facts. Read the storage schema and daemon call path before interpreting rows. Normal storage initialization and local migrations are acceptable. Preserve the evidence used in the case so later experiments can refer to it.
- Trace the actual path through hooks, sessions, attention, behavior, application, delivery, and renderer code as needed. Check which event occurred, which condition selected it, and what the renderer acknowledged. Treat missing logs as an evidence limit, especially across restart or rotation.

## Test the explanation

Write a case specific Python script under `.cache/triage/<case-id>/`. Use `pet_daemon.replay.ReplayHarness` to drive the application with controlled hooks, renderer packets, process evidence, and clock advances. It runs the daemon's application logic with isolated storage and records model calls, emitted packets, and events. Use the current API in `pet_daemon/replay.py`; pass the incident's config or pack when asset behavior matters. For a model quality question, run one live model sample by default when credentials are available and record the model, config, prompt, output, and usage. Label fixture responses and other offline stubs clearly; they establish control-flow behavior but cannot establish model quality.

Start replay from the same state for each candidate explanation. Run before and after experiments with identical inputs, timing, config, model mode, and initial storage. Change only the condition under test. A SQLite backup can supply identical initial storage. Set `wall_time` to the scenario's time when history age or retention matters. Preserve the inputs and raw results for both runs. Compare behavior fields, since delivery UUIDs and measured model latency can differ. State which inputs were captured and which were reconstructed. If the available evidence cannot reproduce the symptom, identify the missing input and report the narrower result.

The fixture command in `pet_daemon.replay` supplies commentary reasons directly. Use `ReplayHarness.advance` for automatic scheduling and delivery timing. For a narrower experiment, import the relevant production builder or controller and call it directly.

## Leave a reviewable result

Write `.cache/triage/<case-id>/report.md` and `case.json`. Use [the inspector template](assets/report.md) as a starting point. Adapt the sections to the behavior and remove unused rows.

Make the Markdown read like an inspector UI. Use short labels, status fields, compact tables, timelines, quoted output, and checkboxes. Tell the complete causal story through the sequence of evidence. Keep explanatory text in short finding cells. Put full prompts, raw JSON, code, and detailed assumptions behind `<details>` sections or links to adjacent files. Prefer an input → state → decision → visible result trace over narrative paragraphs. Place uncertainty beside the affected finding. Compare recorded behavior and experimental results explicitly, with each experiment's changed input and outcome visible together.

`case.json` records the scenario, provenance, code/config/model metadata, and artifact paths in fields that fit the case.

Use `pet_daemon.triage.save_case` after each investigation pass. It keeps the current report and case at the case root, and saves numbered `iterations/<number>/report.md`, `case.json`, `result.json`, and the Python script when supplied. Keep raw requests, responses, packets, and SQL results in the iteration result or adjacent files when they matter. A later pass adds an iteration rather than rewriting earlier evidence. Summarize the finding and point the user to the report.
