from typing import TYPE_CHECKING, cast

from mcp.server.auth.middleware.auth_context import get_access_token

from .mcp import Source, context_priority
from .mcp_work import CompanionContext, Thread, first_goals, thread_summary
from .workstream_marks import MarkField, MarkPatch, MarkStore, is_suppressed, mark_rank

if TYPE_CHECKING:
    from .mcp import WorkContext


def mark_values(mark: dict[str, object]) -> dict[str, object]:
    return cast(dict[str, object], mark.get("values", {}))


def marked_summary(
    thread: Thread, limit: int, mark: dict[str, object]
) -> dict[str, object]:
    summary = thread_summary(thread, limit)
    goals = first_goals(thread.reports)
    original = goals[0] if goals else None
    excerpt = cast(dict[str, object], original or summary["summary"])
    text = str(excerpt["text"])
    title = " ".join(text.split())[:160]
    generated = {
        "basis": "derived",
        "author": "simple-pet",
        "source": excerpt["source"],
    }
    values = mark_values(mark)
    fields = cast(dict[str, dict[str, object]], mark.get("fields", {}))
    return {
        **summary,
        "mark": mark,
        "title": values.get("title", title),
        "title_provenance": fields["title"]["provenance"]
        if "title" in fields
        else generated,
        "generated_title": title,
        "effective_goal": {
            "text": values["goal"],
            "provenance": fields["goal"]["provenance"],
        }
        if "goal" in fields
        else original,
    }


def selected_context(
    threads: list[Thread], marks: dict[str, dict[str, object]], limit: int, now: float
) -> tuple[list[Source], list[dict[str, object]], bool]:
    eligible = [
        thread for thread in threads if not is_suppressed(marks.get(thread.id, {}), now)
    ]
    selected: list[Source] = []
    selected_threads: list[Thread] = []
    for priority in range(3):
        groups = []
        for thread in eligible:
            sources = sorted(
                (
                    source
                    for source in (*thread.reports, *thread.observations)
                    if context_priority(source, now) == priority
                ),
                key=lambda source: (
                    not source.id.startswith("report:"),
                    -source.changed_at,
                    source.id,
                ),
            )
            if sources:
                groups.append((thread, sources))
        groups.sort(
            key=lambda group: (
                mark_rank(marks.get(group[0].id, {})),
                not group[1][0].id.startswith("report:"),
                -group[1][0].changed_at,
            )
        )
        while any(sources for _, sources in groups) and len(selected) < limit:
            for thread, sources in groups:
                if not sources or len(selected) == limit:
                    continue
                selected.append(sources.pop(0))
                if thread not in selected_threads:
                    selected_threads.append(thread)
        if len(selected) == limit:
            break
    summaries = [
        marked_summary(thread, 1, marks.get(thread.id, {}))
        for thread in selected_threads
    ]
    total = sum(len(thread.reports) + len(thread.observations) for thread in eligible)
    return selected, summaries, total > limit


class WorkstreamMarks:
    def __init__(self, context: WorkContext) -> None:
        self.context = context
        self.work = CompanionContext(context)
        self.store = MarkStore(context.path, context.pet, context.clock)

    def require_thread(self, id: str) -> None:
        if not any(thread.id == id for thread in self.work.load_threads()):
            raise ValueError("Unknown workstream ID")

    def mark(
        self, id: str, patch: MarkPatch, author: str, reason: str
    ) -> dict[str, object]:
        self.require_thread(id)
        token = get_access_token()
        principal = f"bearer:{token.client_id}" if token else "local-stdio"
        event = self.store.append(
            id, patch, author=author, reason=reason, principal=principal
        )
        return {
            "pet": self.context.pet,
            "workstream_id": id,
            "event": event,
            "mark": self.store.effective(id),
        }

    def unmark(
        self, id: str, fields: list[MarkField] | None, author: str, reason: str
    ) -> dict[str, object]:
        if not self.store.history(id, limit=1):
            self.require_thread(id)
        token = get_access_token()
        principal = f"bearer:{token.client_id}" if token else "local-stdio"
        event = self.store.unmark(
            id, fields=fields, author=author, reason=reason, principal=principal
        )
        return {
            "pet": self.context.pet,
            "workstream_id": id,
            "event": event,
            "mark": self.store.effective(id),
        }

    def history(
        self, id: str, limit: int = 20, before: int | None = None
    ) -> dict[str, object]:
        events = self.store.history(id, limit=limit + 1, before=before)
        return {
            "pet": self.context.pet,
            "workstream_id": id,
            "checked_at": self.context.clock(),
            "events": events[:limit],
            "truncated": len(events) > limit,
            "next_before": events[limit - 1]["id"] if len(events) > limit else None,
        }
