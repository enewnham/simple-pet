import json
import unicodedata
import uuid
from collections.abc import Sequence

from .attention import AttentionSelection
from .domain import ExecutionOutcome, SpeechReason, SpeechSituation
from .identity import SessionIdentity
from .occasion import OCCASION_SECONDS, Occasion, occasion_fingerprint
from .prose import work_prose
from .runtime import AgentSessionStatus, ReactionKind, ReactionTrigger, Trigger
from .session_notes import SessionNotes
from .sessions.tracking import Sessions
from .storage.memory import Memory
from .work.descriptions import describe_occasion, short_report
from .work.history import report_prose, report_source_key
from .work.models import (
    ExecutionEvidence,
    IntentEvidence,
    ProgressEvidence,
    SessionReport,
)
from .workstream_marks import thread_id

REACTIONS = {
    ReactionKind.START: (SpeechSituation.PROMPT, "A new work turn started."),
    ReactionKind.WAITING: (
        SpeechSituation.CHECK_IN_WAITING,
        "Work is waiting for input.",
    ),
    ReactionKind.TROUBLE: (
        SpeechSituation.STRUGGLE,
        "The same kind of operation has failed repeatedly.",
    ),
    ReactionKind.RECOVERY: (
        SpeechSituation.RECOVERY,
        "An operation recovered after earlier failures.",
    ),
    ReactionKind.FAILURE: (SpeechSituation.FAILURE, "An execution failed."),
    ReactionKind.CANCELLED: (
        SpeechSituation.INTERRUPTED,
        "An execution was cancelled.",
    ),
    ReactionKind.SUCCESS: (
        SpeechSituation.COMPLETION,
        "An execution finished successfully.",
    ),
}
CHECK_INS = {
    SpeechSituation.CHECK_IN_RUNNING: "A tool call is still running.",
    SpeechSituation.CHECK_IN_WAITING: "The work is waiting for input.",
    SpeechSituation.CHECK_IN_QUIET: "No recent activity has been observed in the open work.",
}


def normalized_report(text: str) -> str:
    words = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in words).split())


def related_work(report: SessionReport) -> str | SessionIdentity | None:
    return report.context.pull_request or report.parent


def completed_successfully(report: SessionReport) -> bool:
    return (
        isinstance(report.payload, ExecutionEvidence)
        and report.payload.event in {"turn_done", "subagent_done"}
        and report.payload.outcome == ExecutionOutcome.SUCCESS
    )


def selected_reports(
    reports: Sequence[SessionReport], observed_at: float
) -> tuple[SessionReport, ...]:
    matching = tuple(
        report
        for report in reports
        if report.observed_at == observed_at
        and (
            isinstance(report.payload, ProgressEvidence)
            or isinstance(report.payload, ExecutionEvidence)
            and report.payload.event
            in {"turn_done", "turn_error", "subagent_done", "interrupt"}
        )
    )
    if not matching:
        return ()
    text = report_prose(matching[-1])
    return tuple(report for report in matching if report_prose(report) == text)


class Occasions:
    def __init__(self, memory: Memory, sessions: Sessions) -> None:
        self.memory = memory
        self.sessions = sessions

    @staticmethod
    def session_thread(identity: SessionIdentity, checkout: str) -> str:
        key = checkout or json.dumps(identity.model_dump(mode="json"), sort_keys=True)
        return thread_id(key)

    def character_for(
        self, selection: AttentionSelection, notes: SessionNotes
    ) -> str | None:
        focus = selection.focus_session
        if focus is None or not notes.notes:
            return None
        update = selection.update
        when = update.observed_at if update else focus.last_hook_at
        checkouts = {focus.repo}
        if update is not None:
            reports = self.memory.narrative_reports((update.session,))
            checkouts = {
                report.context.checkout
                for report in selected_reports(reports, when)
                if report.context.checkout
            } or checkouts
        workstreams = {
            self.session_thread(focus.id, checkout) for checkout in checkouts
        }
        assignment = notes.assignment(focus.id, workstreams, when)
        if assignment is None:
            return None
        actor = assignment.actor
        description = [actor.name]
        if actor.description:
            description.append(actor.description)
        if assignment.acting_for:
            description.append(f"Acting for {assignment.acting_for.name}")
        return ". ".join(description)

    def selected_sources(self, selection: AttentionSelection) -> tuple[str, ...]:
        if selection.memory is not None:
            return (selection.memory.key,)
        if selection.update is None or selection.situation in (
            SpeechSituation.PROMPT,
            SpeechSituation.FAILURE,
        ):
            return ()
        reports = self.memory.narrative_reports((selection.update.session,))
        return tuple(
            report_source_key(report)
            for report in selected_reports(reports, selection.update.observed_at)
        )

    def shared_success(
        self, session: SessionIdentity, result: str | None, now: float
    ) -> Occasion | None:
        text = normalized_report(work_prose(result, 280))
        if not text:
            return None
        current = next(
            (
                report
                for report in self.memory.latest_session_reports(session)
                if completed_successfully(report)
                and report.observed_at <= now
                and normalized_report(work_prose(report_prose(report), 280)) == text
            ),
            None,
        )
        scope = related_work(current) if current else None
        parent = self.sessions.by_id[session].parent
        if scope is None:
            scope = parent or session
        if current is None:
            return None
        previous = next(
            (
                report
                for report in reversed(self.memory.recent_work_reports(64))
                if self.sessions.aliases.get(
                    report.payload.session, report.payload.session
                )
                != session
                and now - OCCASION_SECONDS <= report.observed_at <= now
                and self.shares_result(report, scope, text)
            ),
            None,
        )
        if previous is None:
            return None
        return Occasion(
            fingerprint=occasion_fingerprint("result", scope, text),
            fact="Two agents reported the same successful result for shared work.",
            situation=SpeechSituation.SHARED_SUCCESS,
            expires_at=now + OCCASION_SECONDS,
            sources=(report_source_key(previous), report_source_key(current)),
        )

    def shares_result(
        self, report: SessionReport, scope: str | SessionIdentity, text: str
    ) -> bool:
        parent = self.sessions.by_id.get(report.payload.session)
        related = related_work(report) or (parent.parent if parent else None)
        return (
            (
                related == scope
                or (
                    isinstance(scope, SessionIdentity)
                    and report.payload.session == scope
                )
            )
            and completed_successfully(report)
            and normalized_report(work_prose(report_prose(report), 280)) == text
        )

    def select(
        self, selection: AttentionSelection, trigger: Trigger | None
    ) -> Occasion | None:
        now = self.memory.clock()
        if selection.reason == SpeechReason.CARE:
            return Occasion(
                fingerprint=uuid.uuid4().hex,
                fact="There is a quiet moment.",
                situation=selection.situation,
                expires_at=now + OCCASION_SECONDS,
            )
        if selection.id is None:
            return None
        focus = selection.focus_session
        reaction = trigger.reaction if isinstance(trigger, ReactionTrigger) else None
        description = REACTIONS.get(reaction.kind) if reaction else None
        situation, fact = description or (
            selection.situation,
            CHECK_INS.get(selection.situation),
        )
        if focus and situation in {
            SpeechSituation.CHECK_IN_WAITING,
            SpeechSituation.CHECK_IN_RUNNING,
        }:
            return self.waiting(focus, situation, fact, now)
        if focus:
            fact = describe_occasion(situation, focus, fact)
        return self.work_update(selection, situation, fact, now)

    def waiting(
        self,
        focus: AgentSessionStatus,
        situation: SpeechSituation,
        fact: str | None,
        now: float,
    ) -> Occasion | None:
        session = self.sessions.by_id.get(focus.id)
        identity = session.turn if session and session.turn else focus.generation
        if situation == SpeechSituation.CHECK_IN_RUNNING:
            tool = max(
                focus.active_tools, key=lambda item: item.duration_seconds, default=None
            )
            if tool is None or focus.activity_uncertain:
                return None
            key = (identity, tool.call)
        else:
            if not focus.waiting_for_input:
                return None
            key = (
                identity,
                session.waiting_call if session else None,
                focus.waiting_kind,
                focus.waiting_prompt,
            )
        return Occasion(
            fingerprint=occasion_fingerprint(situation, focus.id, key),
            fact=describe_occasion(situation, focus, fact),
            situation=situation,
            expires_at=now + OCCASION_SECONDS,
        )

    def work_update(
        self,
        selection: AttentionSelection,
        situation: SpeechSituation,
        fact: str | None,
        now: float,
    ) -> Occasion | None:
        focus = selection.focus_session
        update = selection.update
        report_session = update.session if update else focus.id if focus else None
        reports = (
            self.memory.narrative_reports((report_session,)) if report_session else ()
        )
        observed_at = (
            now
            if selection.reason == SpeechReason.CHECK_IN
            else update.observed_at
            if update
            else focus.last_hook_at
            if focus
            else now
        )
        picked = selected_reports(reports, update.observed_at) if update else ()
        if situation == SpeechSituation.PROMPT:
            picked = tuple(
                report
                for report in reports
                if isinstance(report.payload, IntentEvidence)
            )[-1:]
        if situation == SpeechSituation.FAILURE:
            picked = ()
        result = picked[-1] if picked else None
        sources = tuple(report_source_key(report) for report in picked)
        if result and not isinstance(result.payload, IntentEvidence):
            text = normalized_report(work_prose(report_prose(result), 280))
            scope = (
                related_work(result)
                or (selection.subject.parent if selection.subject else None)
                or selection.id
            )
            parts = (
                ("result", scope, text)
                if text
                else (
                    situation,
                    selection.id,
                    update.revision if update else observed_at,
                )
            )
        else:
            parts = (
                situation,
                selection.id,
                update.revision
                if update
                else focus.generation
                if focus
                else observed_at,
            )
        if result and situation in {
            SpeechSituation.ROUTINE,
            SpeechSituation.PROMPT,
            SpeechSituation.COMPLETION,
        }:
            fact = (
                short_report(picked, result.observed_at)
                if not selection.subject or not selection.subject.story.intent
                else None
            )
        if fact is None and not sources:
            if focus is None or not focus.turn_active:
                return None
            fact = "An agent is working."
        return Occasion(
            fingerprint=occasion_fingerprint(*parts),
            fact=fact,
            situation=situation,
            expires_at=observed_at + OCCASION_SECONDS,
            sources=sources,
        )
