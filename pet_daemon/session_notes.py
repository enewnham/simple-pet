from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter

from .domain import SpeechSituation
from .identity import SessionIdentity
from .occasion import Occasion, occasion_fingerprint
from .workstream_marks import (
    SESSION_NOTE_LIMIT,
    Assignment,
    Handoff,
    Launch,
    MarkStore,
    Presence,
    Wait,
    WorkstreamNote,
)

NOTES = TypeAdapter(tuple[WorkstreamNote, ...])
NOTE_REACTION_SECONDS = 300


class SessionNotes:
    def __init__(self, path: Path, pet: str, clock: Callable[[], float]) -> None:
        self.store = MarkStore(path, pet, clock)
        self.presence: Presence | None = None
        self.last_note_id: int | None = None
        self.notes: dict[str, dict[str, tuple[float, tuple[WorkstreamNote, ...]]]] = {}

    def refresh(self) -> None:
        presence = self.store.latest_presence()
        self.presence = (
            Presence.model_validate(presence["values"]) if presence else None
        )
        for event in self.store.note_updates(self.last_note_id):
            id = cast(str, event["workstream_id"])
            principal = cast(str, event["principal"])
            values = cast(dict[str, object], event["values"])
            notes = NOTES.validate_python(values.get("notes", []))
            writers = self.notes.pop(id, {})
            if notes:
                writers[principal] = cast(float, event["timestamp"]), notes
            else:
                writers.pop(principal, None)
            if writers:
                self.notes[id] = writers
            if len(self.notes) > SESSION_NOTE_LIMIT:
                self.notes.pop(next(iter(self.notes)))
            self.last_note_id = cast(int, event["id"])

    def human_prompt(self, session: SessionIdentity, turn: str | None) -> bool:
        presence = self.presence
        return bool(
            turn is not None
            and presence
            and presence.turn == turn
            and presence.session
            and presence.session.identity() == session
        )

    def assignment(
        self,
        session: SessionIdentity,
        workstream_ids: Iterable[str],
        observed_at: float | None = None,
    ) -> Assignment | None:
        assignments = {
            note
            for _, at, note in self.for_work(workstream_ids)
            if isinstance(note, Assignment)
            and note.session.identity() == session
            and (observed_at is None or at <= observed_at)
        }
        return next(iter(assignments)) if len(assignments) == 1 else None

    def for_work(
        self, workstream_ids: Iterable[str]
    ) -> Iterable[tuple[str, float, WorkstreamNote]]:
        for id in sorted(set(workstream_ids)):
            for at, notes in self.notes.get(id, {}).values():
                for note in notes:
                    yield id, at, note

    def occasions(
        self, workstream_ids: Iterable[str], now: float
    ) -> tuple[Occasion, ...]:
        result = []
        for id, at, note in self.for_work(workstream_ids):
            expires_at = at + NOTE_REACTION_SECONDS
            if now >= expires_at:
                continue
            match note:
                case Assignment():
                    continue
                case Handoff(sender=sender, to=recipient, work=work):
                    fact = f"{sender.name} passed {work} to {recipient.name}."
                    situation = SpeechSituation.HANDOFF
                case Wait(actor=actor, on=on, detail=detail):
                    fact = f"{actor.name} is waiting for {on.name} on {detail}."
                    situation = SpeechSituation.EXTERNAL_WAIT
                case Launch(by=by):
                    fact = f"{by.name} started another agent session for the work."
                    situation = SpeechSituation.CONTRIBUTOR
            result.append(
                Occasion(
                    fingerprint=occasion_fingerprint("note", id, situation, fact),
                    fact=f"An outside tool reports: {fact}",
                    situation=situation,
                    expires_at=expires_at,
                )
            )
        return tuple(result)
