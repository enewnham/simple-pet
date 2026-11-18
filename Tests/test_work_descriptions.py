from pet_daemon.domain import SpeechSituation
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import WaitingKind
from pet_daemon.work.descriptions import describe_occasion, short_report
from pet_daemon.work.models import ProgressEvidence, ReportContext, SessionReport
from Tests.fixtures import session_snapshot


def test_short_report_uses_compact_prose_from_the_selected_observation() -> None:
    session = SessionIdentity(source="codex", native_id="parser")
    context = ReportContext()

    def report(at: float, text: str) -> SessionReport:
        return SessionReport(
            context=context,
            observed_at=at,
            payload=ProgressEvidence(session=session, text=text),
        )

    reports = (
        report(1, "Fixed the parser."),
        report(2, "Added a regression test."),
        report(2, "Reviewed the failure.\n\nRecorded the details."),
        report(3, "Revised the release notes."),
    )
    assert short_report(reports, 2) == "The agent reported: Added a regression test."
    assert short_report(reports, 4) is None


def test_wait_description_retains_the_request_and_elapsed_time() -> None:
    focus = session_snapshot(
        waiting_for_input=True,
        waiting_kind=WaitingKind.QUESTION,
        waiting_prompt="Which parser should I use?",
        waiting_seconds=120,
    )
    assert describe_occasion(SpeechSituation.CHECK_IN_WAITING, focus, None) == (
        "The work is waiting for an answer to a question. "
        "The request is: Which parser should I use? It has been open for 2 minutes."
    )
