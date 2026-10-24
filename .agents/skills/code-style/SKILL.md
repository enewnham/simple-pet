---
name: code-style
description: Apply Simple Pet's code conventions when implementing, refactoring, or reviewing Python, Swift, or integration scripts. Use for choices about naming, representation, decomposition, state, boundaries, and effects.
---

# Code style

Follow these preferences in code you touch. Use the surrounding implementation to
resolve details. Apply them in proportion to the change, without expanding a task
into a repository cleanup. Formatting, lint rules, type correctness, and the
architecture described in AGENTS.md have their own sources.

Read [references/examples.md](references/examples.md) when a convention needs a
concrete example or when deciding whether an existing pattern fits new code.

## Name the meaning and responsibility

- Name classes for the concepts they represent or responsibilities they own.
  Name files and modules for the cohesive subject or responsibility they contain.
  A name should explain the code without requiring knowledge of its processing
  stage or representation. Avoid `View`, `Projection`, `Presented`, or `Prepared`
  when they only describe how another object was transformed.
- Read a name in its namespace. Keep conventional names such as `Candidate`,
  `Policy`, and `Cognition` when the surrounding module supplies their meaning.
  Add qualification to resolve ambiguity, rather than repeating context.
  Keep familiar names such as `Context` when they describe the role accurately.
  Prefer the shortest clear name over a description of all its fields or callers.
- Name functions for the fact they return or the action they perform. Expose
  meaningful effects such as generating text, recording history, or updating
  state. Use `mark_` when recording that something happened. Keep query names
  distinct from commands that change state.
- Name boolean query methods and properties as questions. Use `is_` for state
  or classification, `has_` for presence, `can_` for capability or permission,
  and `should_` for a policy recommendation. Use Python snake case and Swift
  equivalents such as `isWorking` and `hasPending`. Commands that return success
  keep action names such as `send` and `acknowledge`. Preserve names required by
  framework APIs and external contracts.
- Examine the responsibility when a name is hard to state. Replacing a generic
  suffix with another generic suffix does not establish a useful concept. Share
  domain records when differences concern only serialization, and define a
  separate type when it represents a distinct contract or meaning.

## Choose vocabulary by ownership

- Use provider-owned strings and string literals at integration boundaries.
  Use enums for closed concepts the daemon owns, such as activity, execution
  outcome, and delivery status. Translate provider vocabulary in the adapter.
  Keep open values such as tool names, model IDs, and sprite clip IDs as strings.
- Use dataclasses in Python and structs in Swift for records with named meaning.
  Prefer frozen dataclasses and immutable fields for observations, decisions, and
  snapshots. Keep mutable records for state an owner actively maintains.
- Keep dictionaries for wire payloads, serialization, and keyed lookups. Give
  internal records domain names rather than passing provider payloads through
  policy code.

## Separate decisions from coordination

- Put selection, scoring, association, and projection rules in functions or small
  policy objects that operate on supplied values. Let coordinators order calls
  and apply their results.
- Give a class a concrete responsibility, such as tracking sessions or delivering
  speech. Extract helpers around a meaningful operation or rule. Avoid generic
  managers and helpers that only forward arguments.
- Pass collaborators directly. Use a callable for a single operation and a small
  protocol when a consumer needs several related operations. Introduce an
  abstraction for a real substitution or boundary, rather than hypothetical reuse.

## Keep local flow easy to read

- Handle irrelevant events and ineligible cases early so the main operation stays
  visible. Break a condition into named values when those values explain a rule.
- Use comprehensions for direct filtering and projection. Use loops when ordering,
  mutation, or intermediate decisions carry the meaning.
- Use tables for repeated mappings and regular policy choices. Keep exceptions
  and ordered decisions in ordinary control flow rather than building a rule
  framework. Name constants when they express a policy or shared limit.

## Give state one owner

- Change related state through the component that owns the transition. Keep the
  decision about accepting an event beside the changes that acceptance causes.
- Derive views and summaries from owned records. Store a separate value when it
  represents history, a deadline, or another fact that cannot be reconstructed
  from current state.
- Pass snapshots across component boundaries. Build a changed value with
  `dataclasses.replace` or a new record when readers need a stable observation.
  Group persistent changes that describe one accepted event in one transaction.

## Normalize at the boundary

- Parse, validate, and translate external input where it enters the application.
  Internal code should work with the resulting contract. Keep encoding and
  presentation transformations explicit at their output boundaries.
- Preserve distinctions the source can express, including absent, empty, false,
  unavailable, and unknown. Choose defaults because the domain defines them.
- Keep source facts separate from interpretations. A report, a repository
  observation, and a renderer acknowledgment establish different facts. Preserve
  the context needed to interpret each one later.

## Make time and asynchronous effects explicit

- Supply clocks, random sources, command runners, model clients, and emitters to
  components whose behavior depends on them. Use monotonic time for elapsed
  durations and wall time for persisted observations. Keep the clock choice clear
  at construction.
- Capture the identity or revision an asynchronous operation concerns. Check it
  before applying a result when intervening events can supersede the request.
- Keep requesting an effect and observing its completion distinct. Match feedback
  to the command or message that caused it. Let the component that creates tasks,
  timers, sockets, or observers own their cancellation and cleanup.

## Recover where the recovery is defined

- Catch expected failures where the caller can choose a meaningful outcome.
  Preserve the cause when translating an exception. Reserve broad catches for
  lifecycle boundaries that must isolate a failed operation.
- Represent a failed observation explicitly when downstream decisions need to
  distinguish failure from absence. Retain prior data only when its age and failed
  refresh remain visible.
- Give retries a single owner and an explicit budget. Let cancellation propagate
  after cleanup. Log failures and outcomes where they are handled, with stable
  event names and structured identifiers that connect the operation to its result.

## Test through observable behavior

- Drive the smallest production entrypoint that proves the behavior. Exercise a
  coordinator when the claim depends on cooperation between components.
- Substitute external effects with small fakes and captured outputs. Prefer real
  domain objects and temporary SQLite storage when their behavior matters to the
  claim. Reuse fixture builders for substantial setup and keep scenario-specific
  events and expectations visible in the test.
- Advance a supplied clock or release a controlled response for timing and race
  cases. Check resulting state, packets, or stored facts. Use elapsed-time waits
  for tests that actually exercise system scheduling or desktop behavior.
