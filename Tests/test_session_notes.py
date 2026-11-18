from pathlib import Path

from pet_daemon.domain import SessionSource, SpeechSituation
from pet_daemon.session_notes import SessionNotes
from pet_daemon.workstream_marks import (
    Actor,
    Assignment,
    Handoff,
    MarkPatch,
    SessionRef,
)


def test_session_notes_require_matching_work_and_do_not_name_older_reports(
    tmp_path: Path,
) -> None:
    notes = SessionNotes(tmp_path / "memory.sqlite3", "fixture", lambda: 100)
    session = SessionRef(source=SessionSource.CODEX, session_id="work")
    assignment = Assignment(
        kind="assignment",
        session=session,
        actor=Actor(name="Reviewer", description="Checks repository changes"),
        acting_for=Actor(name="Release lead"),
    )
    notes.store.append(
        "workstream",
        MarkPatch(notes=(assignment,)),
        "coordinator",
        "Assign review",
    )
    notes.refresh()
    assert notes.assignment(session.identity(), ("workstream",), 100) == assignment
    assert notes.assignment(session.identity(), ("workstream",), 99) is None
    assert notes.assignment(session.identity(), ("other-workstream",)) is None


def test_handoff_note_uses_one_fingerprint_across_retries_and_disappears_on_clear(
    tmp_path: Path,
) -> None:
    now = 100.0
    notes = SessionNotes(tmp_path / "memory.sqlite3", "fixture", lambda: now)
    note = Handoff(
        kind="handoff",
        sender=Actor(name="Writer"),
        to=Actor(name="Editor"),
        work="the draft",
    )
    for _ in range(2):
        notes.store.append(
            "workstream", MarkPatch(notes=(note,)), "coordinator", "Moved work"
        )
        notes.refresh()
        (occasion,) = notes.occasions(("workstream",), now)
        if now == 100:
            fingerprint = occasion.fingerprint
        assert occasion.fingerprint == fingerprint
        assert occasion.situation is SpeechSituation.HANDOFF
        assert (
            occasion.fact
            == "An outside tool reports: Writer passed the draft to Editor."
        )
        now += 1

    assert not notes.occasions(("workstream",), now + 300)
    notes.store.unmark("workstream", "coordinator", "Clear handoff", ["notes"])
    notes.refresh()
    assert not notes.occasions(("workstream",), now)


def test_prompt_presence_is_scoped_to_one_turn(tmp_path: Path) -> None:
    notes = SessionNotes(tmp_path / "memory.sqlite3", "fixture", lambda: 100)
    session = SessionRef(source=SessionSource.CODEX, session_id="work")
    notes.store.record_presence(
        "available", "coordinator", "Saw a human type", session=session, turn="turn-1"
    )
    notes.refresh()
    assert notes.human_prompt(session.identity(), "turn-1")
    assert not notes.human_prompt(session.identity(), "turn-2")
    assert not notes.human_prompt(session.identity(), None)


def test_session_notes_keep_both_writers_on_startup_and_refresh(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    notes = SessionNotes(path, "fixture", lambda: 100)
    note = Handoff(
        kind="handoff",
        sender=Actor(name="Builder"),
        to=Actor(name="Reviewer"),
        work="the build",
    )
    review = note.model_copy(update={"work": "the review"})
    for writer, value in (("first", note), ("second", review)):
        notes.store.append(
            "workstream",
            MarkPatch(notes=(value,)),
            "caller",
            "Handoff",
            principal=writer,
        )
    notes.refresh()
    assert {note for _, _, note in notes.for_work(("workstream",))} == {note, review}

    notes.store.unmark("workstream", "caller", "Clear", principal="first")
    restarted = SessionNotes(path, "fixture", lambda: 100)
    for reader in (notes, restarted):
        reader.refresh()
        assert list(reader.for_work(("workstream",))) == [("workstream", 100, review)]
