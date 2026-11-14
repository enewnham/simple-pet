import json
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from .domain import PullRequestState
from .mcp import COVERAGE, Passage, Source, WorkContext, digest
from .mcp_sessions import LiveSession, read_snapshot
from .mcp_work import (
    CompanionContext,
    Thread,
    question_candidates,
    report_activity_at,
    thread_id,
)

if TYPE_CHECKING:
    from .mcp_sessions import SessionSnapshot

from .workstream_marks import is_suppressed, mark_rank

STALL_SECONDS = 24 * 60 * 60


class SessionInfo(TypedDict):
    id: str
    session: dict[str, object]
    workstream_id: str
    checkout: str
    state: str
    observed_at: float
    last_activity_at: float
    presence_checked_at: float
    activity_uncertain: bool
    waiting_kind: str | None
    waiting_prompt: str | None
    sources: list[Passage]
    truncated: bool


class SessionSelection(TypedDict):
    pet: str
    checked_at: float
    coverage: str
    live_available: bool
    snapshot_at: float | None
    truncated: bool
    sessions: list[SessionInfo]


class AttentionItem(TypedDict):
    mark: NotRequired[dict[str, object]]
    title: NotRequired[str]
    title_provenance: NotRequired[dict[str, object]]
    kind: str
    workstream_id: str
    session: dict[str, object] | None
    reason: str
    candidate: bool
    changed_at: float
    sources: list[Passage]


class AttentionSelection(TypedDict):
    pet: str
    checked_at: float
    coverage: str
    live_available: bool
    snapshot_at: float | None
    truncated: bool
    items: list[AttentionItem]


def session_source(session: LiveSession) -> Source:
    identity = session.id.model_dump(mode="json")
    data = session.model_dump_json()
    return Source(
        "session:" + digest(json.dumps(identity, sort_keys=True)),
        data,
        digest(data),
        session.observed_at,
        {
            "basis": "observed",
            "source": "daemon_runtime",
            "session": identity,
            "observed_at": session.observed_at,
            "presence": session.presence,
        },
    )


def session_key(session: LiveSession, threads: list[Thread]) -> str:
    checkouts = [
        thread.checkout
        for thread in threads
        if thread.checkout
        and (
            session.checkout == thread.checkout
            or session.checkout.startswith(thread.checkout.rstrip("/") + "/")
        )
    ]
    if checkouts:
        return max(checkouts, key=len)
    return session.checkout or json.dumps(
        session.id.model_dump(mode="json"), sort_keys=True
    )


def captured_context(session: LiveSession, threads: list[Thread]) -> list[Source]:
    identity = session.id.model_dump(mode="json")
    reports = [
        source
        for thread in threads
        for source in thread.reports
        if source.provenance["session"] == identity
    ]
    observations = [
        source
        for thread in threads
        if thread.key == session_key(session, threads)
        for source in thread.observations
    ]
    return sorted(
        reports,
        key=lambda source: cast(float, source.provenance["observed_at"]),
        reverse=True,
    ) + sorted(observations, key=lambda source: source.changed_at, reverse=True)


def session_info(session: LiveSession, threads: list[Thread]) -> SessionInfo:
    sources = captured_context(session, threads)
    return {
        "id": session_source(session).id,
        "session": session.id.model_dump(mode="json"),
        "workstream_id": thread_id(session_key(session, threads)),
        "checkout": session.checkout,
        "state": session.state,
        "observed_at": session.observed_at,
        "last_activity_at": session.last_activity_at,
        "presence_checked_at": session.presence_checked_at,
        "activity_uncertain": session.activity_uncertain,
        "waiting_kind": session.waiting_kind,
        "waiting_prompt": session.waiting_prompt,
        "sources": [session_source(session).passage()]
        + [source.passage() for source in sources[:4]],
        "truncated": len(sources) > 4,
    }


def live_attention(session: LiveSession, threads: list[Thread]) -> AttentionItem | None:
    if session.state == "waiting_on_permission":
        kind, reason = (
            "permission_pending",
            "The daemon observed a pending permission request.",
        )
    elif session.state == "idle" and session.waiting_kind == "question":
        kind, reason = (
            "unanswered_question",
            "The daemon observed this session waiting for an answer.",
        )
    else:
        return None
    return {
        "kind": kind,
        "workstream_id": thread_id(session_key(session, threads)),
        "session": session.id.model_dump(mode="json"),
        "reason": reason,
        "candidate": False,
        "changed_at": session.last_activity_at,
        "sources": [session_source(session).passage()],
    }


def idle_question(session: LiveSession, threads: list[Thread]) -> AttentionItem | None:
    if session.state != "idle" or session.waiting_kind == "question":
        return None
    reports = tuple(
        source
        for source in captured_context(session, threads)
        if source.id.startswith("report:")
    )
    questions = question_candidates(reports)
    if not questions:
        return None
    source = questions[0]["source"]
    assert isinstance(source, dict)
    return {
        "kind": "unanswered_question",
        "workstream_id": thread_id(session_key(session, threads)),
        "session": session.id.model_dump(mode="json"),
        "reason": "An idle session's latest retained assistant report contains a question. It may still need an answer.",
        "candidate": True,
        "changed_at": session.last_activity_at,
        "sources": [session_source(session).passage(), cast(Passage, source)],
    }


def failed_turns(thread: Thread, latest: dict[str, Source]) -> list[AttentionItem]:
    result = []
    for source in thread.reports:
        key = json.dumps(source.provenance["session"], sort_keys=True)
        if latest[key].id != source.id:
            continue
        execution = source.provenance.get("execution")
        if not isinstance(execution, dict):
            continue
        if execution["event"] != "turn_error" and execution["outcome"] != "error":
            continue
        identity = source.provenance["session"]
        assert isinstance(identity, dict)
        result.append(
            AttentionItem(
                kind="failed_turn",
                workstream_id=thread.id,
                session=identity,
                reason="The latest retained turn for this session reports failure.",
                candidate=True,
                changed_at=source.changed_at,
                sources=[source.passage()],
            )
        )
    return result


def stalled_prs(thread: Thread, now: float) -> list[AttentionItem]:
    activity_at = thread.last_activity_at
    if activity_at is None or now - activity_at < STALL_SECONDS:
        return []
    result = []
    seen = set()
    for source in sorted(
        thread.observations, key=lambda source: source.changed_at, reverse=True
    ):
        value = json.loads(source.text)
        pr = value.get("pr")
        if not pr or pr["url"] in seen:
            continue
        seen.add(pr["url"])
        if pr["state"] != PullRequestState.OPEN:
            continue
        latest = thread.reports[0]
        result.append(
            AttentionItem(
                kind="stalled_pr",
                workstream_id=thread.id,
                session=None,
                reason="An observed open PR has no captured thread activity in at least 24 hours. Verify live PR progress.",
                candidate=True,
                changed_at=activity_at,
                sources=[source.passage(), latest.passage()],
            )
        )
    return result


class CompanionSessions:
    def __init__(self, context: WorkContext) -> None:
        self.context = context
        self.work = CompanionContext(context)

    def snapshot(self, now: float) -> SessionSnapshot | None:
        return read_snapshot(self.context.path, self.context.pet, now)

    def sessions(self, limit: int = 20) -> SessionSelection:
        now = self.context.clock()
        snapshot = self.snapshot(now)
        sessions = list(snapshot.sessions) if snapshot else []
        sessions.sort(key=lambda session: session.last_activity_at, reverse=True)
        threads = self.work.load_threads()
        return {
            "pet": self.context.pet,
            "checked_at": now,
            "coverage": COVERAGE,
            "live_available": snapshot is not None,
            "snapshot_at": snapshot.observed_at if snapshot else None,
            "truncated": len(sessions) > limit,
            "sessions": [
                session_info(session, threads) for session in sessions[:limit]
            ],
        }

    def attention(self, limit: int = 20) -> AttentionSelection:
        now = self.context.clock()
        snapshot = self.snapshot(now)
        sessions = snapshot.sessions if snapshot else ()
        threads = self.work.load_threads()
        items = [
            item
            for session in sessions
            if (item := live_attention(session, threads)) is not None
        ]
        items.extend(
            item
            for session in sessions
            if (item := idle_question(session, threads)) is not None
        )
        latest: dict[str, Source] = {}
        reports = sorted(
            (source for thread in threads for source in thread.reports),
            key=report_activity_at,
            reverse=True,
        )
        for source in reports:
            key = json.dumps(source.provenance["session"], sort_keys=True)
            latest.setdefault(key, source)
        recent_runtime = {
            session_key(session, threads)
            for session in sessions
            if now - session.last_activity_at < STALL_SECONDS
        }
        for thread in threads:
            items.extend(failed_turns(thread, latest))
            if thread.key not in recent_runtime:
                items.extend(stalled_prs(thread, now))
        from .mcp_marks import mark_values, marked_summary

        marks = self.work.marks.all_effective()
        for thread in threads:
            mark = marks.get(thread.id, {})
            if mark_values(mark).get("status") == "blocked":
                values = mark_values(mark)
                sources = (*thread.reports, *thread.observations)
                items.append(
                    AttentionItem(
                        kind="marked_blocked",
                        workstream_id=thread.id,
                        session=None,
                        reason=str(
                            values.get("blocked_on")
                            or values.get("note")
                            or "A caller marked this workstream blocked."
                        ),
                        candidate=False,
                        changed_at=cast(float, mark["updated_at"]),
                        sources=[sources[0].passage()],
                        mark=mark,
                    )
                )
        items = [
            item
            for item in items
            if not is_suppressed(marks.get(item["workstream_id"], {}), now)
        ]
        summaries = {
            thread.id: marked_summary(thread, 1, marks.get(thread.id, {}))
            for thread in threads
        }
        for item in items:
            item["mark"] = marks.get(item["workstream_id"], {})
            summary = summaries.get(item["workstream_id"])
            if summary:
                item["title"] = cast(str, summary["title"])
                item["title_provenance"] = cast(
                    dict[str, object], summary["title_provenance"]
                )
        order = {
            "marked_blocked": 0,
            "permission_pending": 0,
            "failed_turn": 1,
            "unanswered_question": 2,
            "stalled_pr": 3,
        }
        items.sort(
            key=lambda item: (
                mark_rank(item.get("mark", {})),
                order[item["kind"]],
                -item["changed_at"],
            )
        )
        return {
            "pet": self.context.pet,
            "checked_at": now,
            "coverage": COVERAGE,
            "live_available": snapshot is not None,
            "snapshot_at": snapshot.observed_at if snapshot else None,
            "truncated": len(items) > limit,
            "items": items[:limit],
        }
