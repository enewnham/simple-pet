from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from .domain import ArtifactKind, PullRequestState
from .mcp import CONTEXT_STALE_SECONDS, COVERAGE, Source, context_groups
from .work.association import references
from .workstream_marks import MarkStore, is_suppressed, mark_rank, thread_id

if TYPE_CHECKING:
    from .mcp import WorkContext

DECISION_PATTERN = re.compile(
    r"\b(decision\s*:|decided\b|we chose\b|we will use\b|settled on\b|"
    r"correction\s*:|actually\b|I meant\b|do not use\b|don't use\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Thread:
    id: str
    key: str
    checkout: str | None
    reports: tuple[Source, ...]
    observations: tuple[Source, ...]

    @property
    def last_activity_at(self) -> float | None:
        return max(
            (report_activity_at(source) for source in self.reports), default=None
        )

    @property
    def changed_at(self) -> float:
        return max(source.changed_at for source in (*self.reports, *self.observations))


def report_activity_at(source: Source) -> float:
    return cast(float, source.provenance["observed_at"])


def session_key(source: Source) -> str:
    return json.dumps(source.provenance["session"], sort_keys=True)


def group_threads(sources: list[Source]) -> list[Thread]:
    groups = context_groups(sources)
    reported_prs: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        if source.id.startswith("report:"):
            context = source.provenance["context"]
            assert isinstance(context, dict)
            if context["pull_request"]:
                reported_prs[context["pull_request"]].add(groups[source.id])
    grouped: dict[str, list[Source]] = defaultdict(list)
    for source in sources:
        key = groups[source.id]
        if not source.id.startswith("report:"):
            value = json.loads(source.text)
            pr = value.get("pr")
            ref = pr["url"] if pr else value.get("ref")
            candidates = reported_prs.get(ref, set())
            if len(candidates) == 1:
                key = next(iter(candidates))
        grouped[key].append(source)
    result = []
    for key, items in grouped.items():
        reports = tuple(
            sorted(
                (source for source in items if source.id.startswith("report:")),
                key=lambda source: (report_activity_at(source), source.id),
                reverse=True,
            )
        )
        observations = tuple(
            sorted(
                (source for source in items if not source.id.startswith("report:")),
                key=lambda source: (source.changed_at, source.id),
                reverse=True,
            )
        )
        checkout = next(
            (
                source.provenance["context"]["checkout"]
                for source in reports
                if isinstance(source.provenance["context"], dict)
                and source.provenance["context"]["checkout"]
            ),
            None,
        )
        if checkout is None:
            checkout = next(
                (
                    source.provenance["worktree"]
                    for source in observations
                    if "worktree" in source.provenance
                ),
                None,
            )
        assert checkout is None or isinstance(checkout, str)
        result.append(Thread(thread_id(key), key, checkout, reports, observations))
    return sorted(result, key=lambda thread: thread.changed_at, reverse=True)


def statement(
    source: Source, text: str | None = None, offset: int = 0
) -> dict[str, object]:
    return {
        "text": source.text if text is None else text,
        "source": source.passage(offset, min(len(text), 1200) if text else 1200),
    }


def candidate_decisions(reports: tuple[Source, ...]) -> list[dict[str, object]]:
    result = []
    for source in reports:
        offset = 0
        for line in source.text.splitlines(keepends=True):
            text = line.strip()
            if DECISION_PATTERN.search(text):
                start = offset + len(line) - len(line.lstrip())
                result.append(
                    {
                        **statement(source, text[:1200], start),
                        "classification": "decision_or_correction_candidate",
                        "where": source.provenance["context"],
                        "when": source.provenance["observed_at"],
                    }
                )
            offset += len(line)
    return result


def question_candidates(reports: tuple[Source, ...]) -> list[dict[str, object]]:
    latest: dict[str, Source] = {}
    for source in reports:
        latest.setdefault(session_key(source), source)
    result = []
    for source in latest.values():
        if source.provenance["kind"] == "intent":
            continue
        offset = 0
        for line in source.text.splitlines(keepends=True):
            text = line.strip()
            if "?" in text:
                start = offset + len(line) - len(line.lstrip())
                result.append(
                    {
                        **statement(source, text[:1200], start),
                        "classification": "unanswered_question_candidate",
                    }
                )
            offset += len(line)
    return result


def first_goals(reports: tuple[Source, ...]) -> list[dict[str, object]]:
    first: dict[str, Source] = {}
    for source in reversed(reports):
        if source.provenance["kind"] == "intent":
            first.setdefault(session_key(source), source)
    return [
        {
            "session": source.provenance["session"],
            **statement(source, source.text[:1200]),
        }
        for source in first.values()
    ]


def metadata(thread: Thread, limit: int) -> dict[str, object]:
    branches: dict[str, dict[str, object]] = {}
    prs: dict[str, dict[str, object]] = {}
    tickets: dict[str, dict[str, object]] = {}
    for source in (*thread.observations, *thread.reports):
        if source.id.startswith("report:"):
            value = source.provenance["context"]
            assert isinstance(value, dict)
            branch = value["branch"]
            pr = value["pull_request"]
            links = value["references"]
            links = [
                *links,
                *(ref.model_dump(mode="json") for ref in references(source.text)),
            ]
        else:
            value = json.loads(source.text)
            local = value.get("local") or {}
            pr_value = value.get("pr") or {}
            branch = (
                value.get("branch")
                or local.get("branch")
                or pr_value.get("head_branch")
            )
            pr = pr_value.get("url")
            links = pr_value.get("related_links", [])
        if branch:
            branches.setdefault(branch, {"value": branch, "source": source.passage()})
        if pr:
            prs.setdefault(pr, {"value": pr, "source": source.passage()})
        for link in links:
            if link["kind"] == ArtifactKind.TICKET:
                tickets.setdefault(
                    link["ref"], {"value": link["ref"], "source": source.passage()}
                )
    return {
        "branches": list(branches.values())[:limit],
        "pull_requests": list(prs.values())[:limit],
        "tickets": list(tickets.values())[:limit],
        "metadata_truncated": any(
            len(values) > limit for values in (branches, prs, tickets)
        ),
    }


def is_active(thread: Thread, now: float) -> bool:
    if thread.last_activity_at is not None:
        return now - thread.last_activity_at <= CONTEXT_STALE_SECONDS
    for source in thread.observations:
        value = json.loads(source.text)
        observed_at = value.get("last_success_at")
        if observed_at is None:
            observed_at = source.changed_at
        pr = value.get("pr")
        if now - observed_at <= CONTEXT_STALE_SECONDS and (
            value.get("kind") == "checkout"
            and value.get("available", True)
            or pr
            and pr["state"] == PullRequestState.OPEN
        ):
            return True
    return False


def thread_summary(thread: Thread, limit: int) -> dict[str, object]:
    summary = thread.reports[0] if thread.reports else thread.observations[0]
    prose = summary.text.lstrip()
    text = prose.splitlines()[0][:240] if thread.reports else "Repository observation"
    offset = len(summary.text) - len(prose) if thread.reports else 0
    activity_source = thread.reports[0].passage() if thread.reports else None
    sessions = list(
        {
            session_key(source): source.provenance["session"]
            for source in thread.reports
        }.values()
    )
    return {
        "id": thread.id,
        "checkout": thread.checkout,
        "sessions": sessions[:limit],
        "sessions_truncated": len(sessions) > limit,
        "last_activity_at": thread.last_activity_at,
        "last_activity_source": activity_source,
        "summary": {
            **statement(summary, text, offset),
            "method": "extractive" if thread.reports else "observation_label",
        },
        **metadata(thread, limit),
    }


class CompanionContext:
    def __init__(self, context: WorkContext) -> None:
        self.context = context
        self.marks = MarkStore(context.path, context.pet, context.clock)

    def load_threads(self) -> list[Thread]:
        with self.context.database() as db:
            reports = self.context.reports(db, 1, balanced=True)
            observations = self.context.observations(db, None)
        return group_threads(reports + observations)

    def envelope(self, checked_at: float, truncated: bool = False) -> dict[str, object]:
        return {
            "pet": self.context.pet,
            "checked_at": checked_at,
            "coverage": COVERAGE,
            "truncated": truncated,
        }

    def workstreams(self, limit: int = 12) -> dict[str, object]:
        from .mcp_marks import mark_values, marked_summary

        now = self.context.clock()
        marks = self.marks.all_effective()
        threads = []
        for thread in self.load_threads():
            mark = marks.get(thread.id, {})
            values = mark_values(mark)
            explicit_active = (
                values.get("status") in {"active", "blocked", "waiting-review"}
                or values.get("priority") == "focus"
            )
            if not is_suppressed(mark, now) and (
                is_active(thread, now) or explicit_active
            ):
                threads.append(thread)
        threads.sort(
            key=lambda thread: (
                mark_rank(marks.get(thread.id, {})),
                thread.last_activity_at is None,
                -(
                    thread.last_activity_at
                    if thread.last_activity_at is not None
                    else thread.changed_at
                ),
            )
        )
        summaries = [
            marked_summary(thread, limit, marks.get(thread.id, {}))
            for thread in threads[:limit]
        ]
        truncated = len(threads) > limit or any(
            summary["metadata_truncated"] or summary["sessions_truncated"]
            for summary in summaries
        )
        return {
            **self.envelope(now, truncated),
            "activity_window_seconds": CONTEXT_STALE_SECONDS,
            "workstreams": summaries,
        }

    def workstream(self, id: str, limit: int = 12) -> dict[str, object]:
        from .mcp_marks import marked_summary

        now = self.context.clock()
        thread = next(
            (thread for thread in self.load_threads() if thread.id == id), None
        )
        if thread is None:
            raise ValueError("Unknown workstream ID")
        goals = first_goals(thread.reports)
        decisions = candidate_decisions(thread.reports)
        questions = question_candidates(thread.reports)
        truncated = any(
            len(items) > limit
            for items in (
                thread.reports,
                thread.observations,
                goals,
                decisions,
                questions,
            )
        )
        summary = marked_summary(thread, limit, self.marks.effective(thread.id))
        truncated = truncated or bool(
            summary["metadata_truncated"] or summary["sessions_truncated"]
        )
        return {
            **self.envelope(now, truncated),
            **summary,
            "goal": summary["effective_goal"],
            "captured_goal": goals[0] if goals else None,
            "session_goals": goals[:limit],
            "recent_turns": [source.passage() for source in thread.reports[:limit]],
            "decisions": decisions[:limit],
            "open_questions": questions[:limit],
            "repository_state": [
                source.passage() for source in thread.observations[:limit]
            ],
        }

    def since(self, ts: float, limit: int = 20) -> dict[str, object]:
        now = self.context.clock()
        changed = []
        truncated = False
        for thread in self.load_threads():
            sources = sorted(
                (
                    source
                    for source in (*thread.reports, *thread.observations)
                    if source.changed_at > ts
                ),
                key=lambda source: (source.changed_at, source.id),
                reverse=True,
            )
            if sources:
                changed.append(
                    {
                        "id": thread.id,
                        "checkout": thread.checkout,
                        "changed_at": sources[0].changed_at,
                        "truncated": len(sources) > limit,
                        "sources": [source.passage() for source in sources[:limit]],
                    }
                )
                truncated = truncated or len(sources) > limit
        changed.sort(key=lambda thread: cast(float, thread["changed_at"]), reverse=True)
        return {
            **self.envelope(now, truncated or len(changed) > limit),
            "after": ts,
            "workstreams": changed[:limit],
        }

    def decisions(self, query: str = "", limit: int = 12) -> dict[str, object]:
        now = self.context.clock()
        result = []
        terms = re.findall(r"\w+", query.casefold())
        for thread in self.load_threads():
            for candidate in candidate_decisions(thread.reports):
                text = candidate["text"]
                assert isinstance(text, str)
                if terms and not all(term in text.casefold() for term in terms):
                    continue
                if query and not terms:
                    continue
                result.append(
                    {
                        "workstream_id": thread.id,
                        "checkout": thread.checkout,
                        **candidate,
                    }
                )
        result.sort(key=lambda item: cast(float, item["when"]), reverse=True)
        return {
            **self.envelope(now, len(result) > limit),
            "classification": "lexical_candidates",
            "decisions": result[:limit],
        }
