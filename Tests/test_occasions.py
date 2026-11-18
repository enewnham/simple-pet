from pathlib import Path

from pet_daemon.attention import AttentionSelection
from pet_daemon.domain import Activity, SpeechReason, SpeechSituation
from pet_daemon.identity import SessionIdentity
from pet_daemon.occasion import (
    OCCASION_SECONDS,
    STRUGGLE_SECONDS,
    Timers,
    direction_changed,
    occasion_fingerprint,
)
from pet_daemon.occasions import Occasions, normalized_report
from pet_daemon.runtime import ActiveTool
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage.memory import Memory
from pet_daemon.work.shared_work import WorkUpdate
from Tests.fixtures import require, session_snapshot


def test_repeated_failures_recover_only_for_the_matching_operation() -> None:
    session = SessionIdentity(source="codex", native_id="work")
    other = SessionIdentity(source="codex", native_id="other")
    timers = Timers().failed(session, Activity.TESTING, "check", 10)
    timers = timers.failed(session, Activity.TESTING, "check", 15)
    restored = Timers.model_validate_json(timers.model_dump_json())

    assert restored.recovered(other, Activity.TESTING, "check", 20) == (timers, None)
    assert restored.recovered(session, Activity.TESTING, "build", 20) == (timers, None)
    at_expiry = 10 + STRUGGLE_SECONDS
    assert restored.recovered(session, Activity.TESTING, "check", at_expiry) == (
        timers,
        None,
    )

    cleared, occasion = restored.recovered(session, Activity.TESTING, "check", 20)
    assert cleared.struggle is None
    recovery = require(occasion)
    assert recovery.situation == SpeechSituation.RECOVERY
    assert recovery.expires_at == 20 + OCCASION_SECONDS
    assert cleared.recovered(session, Activity.TESTING, "check", 21) == (cleared, None)


def test_success_after_a_single_failure_clears_the_timer_without_commentary() -> None:
    session = SessionIdentity(source="codex", native_id="work")
    timers = Timers().failed(session, Activity.BUILDING, "build", 10)
    cleared, occasion = timers.recovered(session, Activity.BUILDING, "build", 11)
    assert cleared.struggle is None
    assert occasion is None


def test_unanswered_bid_has_one_fingerprint_within_its_window() -> None:
    timers = Timers(unanswered_bid_at=100)
    assert timers.ignored(now=129, delay=30) is None
    first = require(timers.ignored(now=130, delay=30))
    second = require(timers.ignored(now=200, delay=30))
    assert first == second
    assert first.situation == SpeechSituation.UNANSWERED
    assert timers.ignored(now=130 + OCCASION_SECONDS, delay=30) is None


def test_same_result_has_one_fingerprint_within_a_piece_of_work() -> None:
    first = normalized_report("Formatting checks PASS.")
    repeated = normalized_report("Formatting checks pass!")
    assert occasion_fingerprint("result", "work-1", first) == occasion_fingerprint(
        "result", "work-1", repeated
    )
    assert occasion_fingerprint("result", "work-1", first) != occasion_fingerprint(
        "result", "work-2", repeated
    )


def test_direction_change_uses_explicit_leading_wording() -> None:
    assert direction_changed("Actually, stop this approach and try the other.")
    assert direction_changed("Let's switch to the earlier draft.")
    assert not direction_changed("The build may switch to the fallback.")


def test_quiet_check_in_changes_fingerprint_only_when_the_work_changes(
    tmp_path: Path,
) -> None:
    now = 120.0
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: now)
    try:
        opportunities = Occasions(memory, Sessions(lambda: now))
        work = session_snapshot(turn_active=True, generation=1)
        selection = AttentionSelection(
            subject=None,
            focus_session=work,
            context_sessions=(),
            reason=SpeechReason.CHECK_IN,
            situation=SpeechSituation.CHECK_IN_QUIET,
        )
        first = require(opportunities.select(selection, None))
        now += 60
        same_work = require(opportunities.select(selection, None))
        assert same_work.fingerprint == first.fingerprint
        later = selection.model_copy(
            update={
                "update": WorkUpdate(session=work.id, revision="next", observed_at=now)
            }
        )
        assert (
            require(opportunities.select(later, None)).fingerprint != first.fingerprint
        )
    finally:
        memory.close()


def test_running_check_in_uses_the_tool_call_as_its_fingerprint(tmp_path: Path) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 120)
    try:
        opportunities = Occasions(memory, Sessions(lambda: 120))
        tool = ActiveTool(
            call="one", tool="check", activity=Activity.TESTING, duration_seconds=60
        )
        work = session_snapshot(turn_active=True, active_tools=(tool,))
        selection = AttentionSelection(
            subject=None,
            focus_session=work,
            context_sessions=(),
            reason=SpeechReason.CHECK_IN,
            situation=SpeechSituation.CHECK_IN_RUNNING,
        )
        first = require(opportunities.select(selection, None))
        longer = tool.model_copy(update={"duration_seconds": 120})
        changed_work = work.model_copy(update={"active_tools": (longer,)})
        same = require(
            opportunities.select(
                selection.model_copy(update={"focus_session": changed_work}), None
            )
        )
        assert first.fact != same.fact
        assert first.fingerprint == same.fingerprint
        other = tool.model_copy(update={"call": "two"})
        other_work = work.model_copy(update={"active_tools": (other,)})
        replace = require(
            opportunities.select(
                selection.model_copy(update={"focus_session": other_work}), None
            )
        )
        assert first.fingerprint != replace.fingerprint
    finally:
        memory.close()
