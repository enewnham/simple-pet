import asyncio
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from .attention import Attention, AttentionSelection, SelectedWorkState
from .commentary import Cognition, CommentaryResult
from .commentary.contracts import ActionOffer
from .commentary.conversation import ActionChoice, Exchange
from .commentary.request import SpeechConditions, commentary_facts
from .config import Config
from .delivery import DeliveryOutcome, SpeechDelivery, SpeechMessage, SpeechPacket
from .domain import DeliveryStatus, SpeechReason, SpeechSituation
from .identity import SessionIdentity
from .logging import get_logger
from .navigation import Navigator, Target, next_morning
from .runtime import OverlayPacket, PetContext
from .storage import Memory
from .types import JSONObject
from .work.history import report_source_key
from .work.models import ReportContext
from .work.narrative import semantic_revision
from .work.shared_work import WorkSubject

log = get_logger("pet-daemon")
ROUND_LIMIT = 8


class ActionKind(StrEnum):
    NAVIGATE = "navigate"
    CONTEXT = "context"
    TOMORROW = "tomorrow"
    HANDLED = "handled"
    MORE = "more"
    ROUNDS = "rounds"
    BACK = "back"
    DISMISS = "dismiss"
    THREAD = "thread"


@dataclass(frozen=True)
class Point:
    source: str
    subject: WorkSubject
    title: str
    text: str
    context: str
    sources: tuple[str, ...]
    observed_at: float
    conditions: SpeechConditions
    feedback_sources: tuple[str, ...]
    responses: tuple[ActionChoice, ...] | None = None


@dataclass(frozen=True)
class Choice:
    kind: ActionKind
    point: Point | None = None
    target: Target | None = None


class Interactions:
    def __init__(
        self,
        config: Config,
        *,
        memory: Memory,
        attention: Attention,
        cognition: Cognition,
        context: Callable[[], PetContext],
        emit: Callable[[OverlayPacket], object],
        more: Callable[[], None],
        rounds: Callable[[], None],
        displayed: Callable[[str], None],
        dismissed: Callable[[], None],
        clock: Callable[[], float],
    ) -> None:
        self.config = config
        self.memory = memory
        self.attention = attention
        self.cognition = cognition
        self.context = context
        self.emit = emit
        self.more = more
        self.rounds = rounds
        self.displayed = displayed
        self.dismissed = dismissed
        self.navigator = Navigator(config.navigation.handlers)
        self.identifier = ""
        self.revision = 0
        self.mode = "thread"
        self.return_mode = "thread"
        self.persistent = False
        self.engaged = False
        self.points: tuple[Point, ...] = ()
        self.selected: Point | None = None
        self.choices: dict[str, Choice] = {}
        self.seen: set[SessionIdentity] = set()
        self.is_more = False
        self.headline: str | None = None
        self.aside: Exchange | None = None
        self.aside_text = ""
        self.task: asyncio.Task[None] | None = None
        self.delivery = SpeechDelivery(
            self._emit_delivery,
            clock=clock,
            valid=self._is_delivery_valid,
            on_outcome=self._delivery_outcome,
        )

    @property
    def is_interacting(self) -> bool:
        return bool(self.identifier) and (self.persistent or self.engaged)

    def _emit_delivery(self, packet: SpeechPacket) -> None:
        self.emit(
            {
                "text": packet["text"],
                "ttl": packet["ttl"],
                "delivery_id": packet["delivery_id"],
                **(
                    {"conversation": packet["conversation"]}
                    if "conversation" in packet
                    else {}
                ),
            }
        )

    def _is_delivery_valid(self, message: SpeechMessage) -> bool:
        conversation = message.conversation
        return (
            message.selection_id == self.identifier
            and conversation is not None
            and conversation.get("revision") == self.revision
            and all(self._is_current(point) for point in self.points)
        )

    def _delivery_outcome(self, outcome: DeliveryOutcome) -> None:
        if outcome.status != DeliveryStatus.ACCEPTED:
            return
        message = outcome.message
        try:
            with self.memory.transaction():
                self.memory.record_acceptance(
                    message.message_id,
                    message.text,
                    message.reason,
                    subject_id=message.subject_id,
                    revision=message.revision,
                )
                self.memory.record_source_mentions(message.mentioned_sources)
            self.attention.restore_sources(self.memory.load_source_attention())
            self.displayed(message.text)
        except sqlite3.Error:
            log.exception("interaction.acceptance_failed")

    def _new(self, persistent: bool) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None
        self.delivery.cancel_pending()
        self.identifier = str(uuid.uuid4())
        self.revision = 0
        self.persistent = persistent
        self.engaged = False
        self.choices.clear()
        self.headline = None
        self.aside = None
        self.aside_text = ""

    def offers(self, selection: AttentionSelection) -> tuple[ActionOffer, ...]:
        if not selection.situation.is_conversation:
            return ()
        offers: dict[str, ActionOffer] = {}
        if selection.subject is not None:
            for report in self.memory.context_reports(selection.subject):
                for target in self.navigator.targets(selection.subject, report.context):
                    offers[target.kind.value] = ActionOffer(
                        target.kind.value,
                        self.navigator.label(target),
                        "Open the recorded "
                        + target.kind.value.replace("_", " ")
                        + " for the cited point.",
                    )
            for identifier, label, description in (
                (
                    "context",
                    "What's the context?",
                    "Show a short explanation of the cited point.",
                ),
                (
                    "tomorrow",
                    "Tomorrow",
                    "Defer this point until tomorrow morning. Useful for an explicitly remaining thread.",
                ),
                (
                    "handled",
                    "Already handled",
                    "Record the developer's correction that this remaining point is already handled.",
                ),
            ):
                offers[identifier] = ActionOffer(identifier, label, description)
        if self.context().work:
            offers["more"] = ActionOffer(
                "more", "Another thread", "Look at another work thread."
            )
            offers["rounds"] = ActionOffer(
                "rounds",
                "Recent work",
                "Show a brief tour of recent work and explicitly remaining threads.",
            )
        return tuple(offers.values())

    def begin(self, *, more: bool = False) -> None:
        if not more:
            self.seen.clear()
        self.is_more = more
        self._new(True)
        self.points = ()
        self.selected = None
        self.mode = "loading"
        self.emit({"conversation": self._snapshot("Let me see where we left things.")})

    def choose(
        self, snapshot: PetContext, selection: AttentionSelection
    ) -> AttentionSelection | None:
        candidates = [
            subject
            for subject in snapshot.work
            if subject.parent is None and subject.id not in self.seen
        ]
        candidates.sort(
            key=lambda subject: (
                subject.id != selection.id,
                -subject.changed_at,
                subject.id,
            )
        )
        for subject in candidates:
            reports = self.memory.narrative_reports((subject.id,))
            if not any(
                self.attention.is_source_available(
                    report_source_key(report), report.observed_at
                )
                for report in reports
            ):
                continue
            if subject.id == selection.id:
                return selection
            focus = next(
                (session for session in snapshot.sessions if session.id == subject.id),
                None,
            )
            return AttentionSelection(
                subject=subject,
                focus_session=focus,
                context_sessions=(),
                reason=SpeechReason.CLICK,
                situation=SpeechSituation.CLICK_WAITING
                if focus and focus.waiting_for_input
                else SpeechSituation.CLICK_WORKING
                if focus and focus.turn_active
                else SpeechSituation.IDLE_REFLECTION,
                selected_state=SelectedWorkState.capture(subject, focus),
            )
        return None if self.is_more or candidates else selection

    def _observed_at(self, source: str, subject: WorkSubject) -> float:
        narrative = self.memory.load_narrative(subject.id)
        record = narrative.sources.get(source) if narrative else None
        if record is not None and isinstance(
            value := record.get("observed_at"), (int, float)
        ):
            return value
        return next(
            (
                report.observed_at
                for report in self.memory.narrative_reports((subject.id,))
                if report_source_key(report) == source
            ),
            subject.changed_at,
        )

    def _targets(self, point: Point) -> tuple[Target, ...]:
        report = next(
            (
                report
                for report in self.memory.context_reports(point.subject)
                if report_source_key(report) == point.source
            ),
            None,
        )
        context = report.context if report else ReportContext()
        identity = report.payload.session if report else point.subject.id
        if report is None:
            narrative = self.memory.load_narrative(point.subject.id)
            source = narrative.sources.get(point.source, {}) if narrative else {}
            if (origin := source.get("origin")) is not None:
                identity = SessionIdentity.model_validate(origin)
        subject = point.subject.model_copy(update={"id": identity})
        return self.navigator.targets(subject, context)

    def _context_text(self, subject: WorkSubject) -> str:
        narrative = self.memory.load_narrative(subject.id)
        if narrative is None:
            return subject.story.intent
        statements = (narrative.intent, *narrative.decisions, *narrative.progress)
        return "\n\n".join(statement.text for statement in statements[:5])

    def _selected_source(
        self, result: CommentaryResult, sources: tuple[str, ...]
    ) -> str | None:
        memory = result.request.selection.memory
        if memory is not None:
            return memory.key
        preferred = result.mentions if result.exchange is not None else sources
        return next(iter(preferred or sources or result.mentions), None)

    def decorate(
        self, result: CommentaryResult, sources: tuple[str, ...], ttl: float
    ) -> JSONObject | None:
        selection = result.request.selection
        subject = selection.subject
        exchange = result.exchange
        if (
            result.speak is None
            or not selection.situation.is_conversation
            or exchange is None
        ):
            return None
        if subject is None:
            return self._decorate_aside(result.speak, exchange, selection.reason, ttl)
        source = self._selected_source(result, sources)
        if source is None:
            return self._decorate_aside(result.speak, exchange, selection.reason, ttl)
        observed_at = self._observed_at(source, subject)
        if not self.attention.is_source_available(source, observed_at):
            return None
        point = Point(
            source,
            subject,
            exchange.headline,
            result.speak,
            exchange.context,
            result.mentions or (source,),
            observed_at,
            result.request.conditions,
            sources if source in sources else (source,),
            exchange.choices,
        )
        persistent = selection.reason == SpeechReason.CLICK
        self._new(persistent)
        self.headline = exchange.headline
        self.points = (point,)
        self.selected = point
        self.seen.add(subject.id)
        self.mode = "thread"
        return self._snapshot(ttl=ttl)

    def _decorate_aside(
        self, text: str, exchange: Exchange, reason: SpeechReason, ttl: float
    ) -> JSONObject:
        self._new(reason == SpeechReason.CLICK)
        self.points, self.selected, self.mode = (), None, "thread"
        self.headline, self.aside, self.aside_text = exchange.headline, exchange, text
        return self._snapshot(ttl=ttl)

    def is_suppressed(self, result: CommentaryResult, sources: tuple[str, ...]) -> bool:
        selection = result.request.selection
        if selection.subject is None:
            return False
        source = self._selected_source(result, sources)
        return source is not None and not self.attention.is_source_available(
            source, self._observed_at(source, selection.subject)
        )

    def finish_empty(self, text: str = "That's the tour for now.") -> None:
        self.mode = "empty"
        self.points = ()
        self.selected = None
        self.persistent = True
        if not self.identifier:
            self._new(True)
        self.emit({"conversation": self._snapshot(text)})

    async def _round(self, subject: WorkSubject, snapshot: PetContext) -> Point | None:
        selection = AttentionSelection(
            subject=subject,
            focus_session=None,
            context_sessions=(),
            reason=SpeechReason.CLICK,
            situation=SpeechSituation.EXPLICIT,
            selected_state=SelectedWorkState.capture(subject, None),
        )
        narrative = await asyncio.wait_for(
            self.cognition.update_work_narrative(snapshot, selection, self.context),
            self.config.request_timeout,
        )
        if narrative is None:
            return None
        statements = (
            *reversed(narrative.progress),
            *reversed(narrative.decisions),
            narrative.intent,
        )
        for statement in statements:
            sources = tuple(
                source for source in statement.sources if source in narrative.sources
            )
            if not sources or any(
                not self.attention.is_source_available(
                    source, self._observed_at(source, subject)
                )
                for source in sources
            ):
                continue
            source = max(sources, key=lambda key: self._observed_at(key, subject))
            attribution = (
                "Last request"
                if narrative.sources[source].get("kind") == "request"
                else "Last report"
            )
            conditions = SpeechConditions(
                selection=selection,
                context=commentary_facts(snapshot, selection, self.memory.clock()),
                narrative_revision=semantic_revision(subject.story, related=False),
                source_scope_revisions={
                    key: revision
                    for key in sources
                    if isinstance(
                        revision := narrative.sources[key].get("scope_revision"), str
                    )
                },
            )
            return Point(
                source,
                subject,
                narrative.intent.text,
                f"{attribution}: {statement.text}",
                self._context_text(subject),
                sources,
                self._observed_at(source, subject),
                conditions,
                sources,
            )
        return None

    async def show_rounds(self) -> None:
        self.begin()
        token = self.identifier
        snapshot = self.context()
        subjects = [subject for subject in snapshot.work if subject.parent is None][
            :ROUND_LIMIT
        ]
        semaphore = asyncio.Semaphore(3)

        async def describe(subject: WorkSubject) -> Point | None:
            async with semaphore:
                return await self._round(subject, snapshot)

        descriptions = await asyncio.gather(
            *(describe(subject) for subject in subjects), return_exceptions=True
        )
        if token != self.identifier:
            return
        points: list[Point] = []
        seen: set[str] = set()
        failed = False
        for point in descriptions:
            if isinstance(point, BaseException):
                failed = True
                log.warning("interaction.thread_unavailable", error=str(point))
                continue
            if point is None or seen.intersection(point.feedback_sources):
                continue
            if self._is_current(point):
                points.append(point)
                seen.update(point.feedback_sources)
                self.seen.add(point.subject.id)
        self.points = tuple(points)
        self.selected = None
        self.mode = "rounds" if points else "empty"
        text = (
            "Here's a quick tour of where we left things."
            if points
            else "No recent threads to tour."
        )
        if failed and not points:
            text = "I couldn't refresh our notes. Try again in a moment."
        snapshot_packet = self._snapshot(
            text,
            error="Some threads couldn't be refreshed." if failed and points else None,
        )
        self.delivery.submit(
            SpeechMessage(
                text=text,
                ttl=self.config.speech_ttl,
                reason=SpeechReason.CLICK,
                selection_id=self.identifier,
                conversation=snapshot_packet,
                mentioned_sources=tuple(
                    dict.fromkeys(
                        source for point in points for source in point.sources
                    )
                ),
            )
        )

    def _action(
        self,
        kind: ActionKind,
        label: str,
        point: Point | None = None,
        target: Target | None = None,
    ) -> JSONObject:
        identifier = str(uuid.uuid4())
        self.choices[identifier] = Choice(kind, point, target)
        return {"id": identifier, "kind": kind.value, "label": label}

    def _item(self, point: Point) -> JSONObject:
        targets = {target.kind.value: target for target in self._targets(point)}
        responses = point.responses
        if responses is None:
            responses = tuple(
                ActionChoice(id=key, label=self.navigator.label(target))
                for key, target in list(targets.items())[:1]
            )
            responses += (ActionChoice(id="context", label="What's the context?"),)
        actions = []
        for response in responses:
            if response.id in ("worktree", "pull_request", "session"):
                if (target := targets.get(response.id)) is not None:
                    actions.append(
                        self._action(ActionKind.NAVIGATE, response.label, point, target)
                    )
            elif response.id != "context" or self.mode != "context":
                kind = ActionKind(response.id)
                actions.append(
                    self._action(
                        kind,
                        response.label,
                        point
                        if kind
                        in (ActionKind.CONTEXT, ActionKind.TOMORROW, ActionKind.HANDLED)
                        else None,
                    )
                )
        stamp = datetime.fromtimestamp(point.observed_at).strftime("%b %d, %H:%M")
        narrative = self.memory.load_narrative(point.subject.id)
        source = narrative.sources.get(point.source, {}) if narrative else {}
        attribution = (
            "User request" if source.get("kind") == "request" else "Agent report"
        )
        context = point.context + f"\n\n{attribution} · {stamp}."
        return {
            "id": point.source,
            "title": point.title,
            "text": point.text,
            "context": context,
            "actions": tuple(actions),
        }

    def _snapshot(
        self, text: str = "", *, ttl: float | None = None, error: str | None = None
    ) -> JSONObject:
        self.revision += 1
        self.choices.clear()
        points = (self.selected,) if self.selected else self.points
        items = [self._item(point) for point in points]
        if self.aside is not None:
            items.append(
                {
                    "id": self.identifier,
                    "title": self.aside.headline,
                    "text": self.aside_text,
                    "context": self.aside.context,
                    "actions": tuple(
                        self._action(ActionKind(choice.id), choice.label)
                        for choice in self.aside.choices
                    ),
                }
            )
        actions = []
        if self.mode == "context":
            actions.append(self._action(ActionKind.BACK, "Back"))
        elif self.mode != "loading":
            if self.mode != "empty":
                actions.append(self._action(ActionKind.MORE, "Another thread"))
            if self.mode != "rounds":
                actions.append(self._action(ActionKind.ROUNDS, "Make the rounds"))
        actions.append(self._action(ActionKind.DISMISS, "Dismiss"))
        return {
            "id": self.identifier,
            "revision": self.revision,
            "mode": self.mode,
            "persistent": self.persistent,
            "ttl": self.config.speech_ttl if ttl is None else ttl,
            "text": text,
            "items": tuple(items),
            "actions": tuple(actions),
            "error": error,
            "headline": self.headline,
        }

    def _publish(self, text: str = "", *, error: str | None = None) -> None:
        self.emit({"conversation": self._snapshot(text, error=error)})

    def _is_current(self, point: Point) -> bool:
        snapshot = self.context()
        return point.conditions.is_valid(
            snapshot
        ) and self.cognition.are_source_scopes_valid(point.conditions)

    def receive(self, packet: Mapping[str, object]) -> bool:
        event = packet.get("event")
        if event == "speech_accepted":
            return self.delivery.acknowledge(packet)
        if event not in (
            "interaction_action",
            "interaction_dismissed",
            "interaction_state",
        ):
            return False
        if packet.get("interaction_id") != self.identifier or not self.identifier:
            if event == "interaction_action":
                if self.identifier:
                    self._publish(
                        error="That conversation changed. Please choose again."
                    )
                else:
                    self.emit({"dismiss_conversation": True})
            return True
        if event == "interaction_state":
            if isinstance(engaged := packet.get("engaged"), bool):
                self.engaged = engaged
            return True
        if (
            type(packet.get("revision")) is not int
            or packet["revision"] != self.revision
        ):
            self._publish(error="That choice changed. Please choose again.")
            return True
        if event == "interaction_dismissed":
            self.clear()
            return True
        identifier = packet.get("action_id")
        choice = self.choices.get(identifier) if isinstance(identifier, str) else None
        if (
            self.task is not None
            and not self.task.done()
            and (choice is None or choice.kind != ActionKind.DISMISS)
        ):
            return True
        if choice is None:
            self._publish(error="That choice isn't available. Please choose again.")
            return True
        self._choose(choice)
        return True

    def _choose(self, choice: Choice) -> None:
        kind, point = choice.kind, choice.point
        self.delivery.cancel_pending()
        if point is not None and not self._is_current(point):
            self._remove(point)
            return
        if kind == ActionKind.DISMISS:
            self.clear()
        elif kind == ActionKind.MORE:
            self.more()
        elif kind == ActionKind.ROUNDS:
            self.rounds()
        elif kind == ActionKind.BACK:
            self.mode = self.return_mode
            if self.mode == "rounds":
                self.selected = None
            self._publish()
        elif kind in (ActionKind.CONTEXT, ActionKind.THREAD) and point is not None:
            self.return_mode = self.mode
            self.selected = point
            self.persistent = True
            self.mode = "context" if kind == ActionKind.CONTEXT else "thread"
            self._publish()
        elif point is not None and kind in (ActionKind.HANDLED, ActionKind.TOMORROW):
            try:
                self.memory.record_source_feedback(
                    point.feedback_sources,
                    "handled" if kind == ActionKind.HANDLED else "tomorrow",
                    self.memory.clock(),
                    deferred_until=next_morning(self.memory.clock())
                    if kind == ActionKind.TOMORROW
                    else None,
                )
                self.attention.restore_sources(self.memory.load_source_attention())
                self._remove(point)
            except sqlite3.Error:
                log.exception("interaction.feedback_failed", source=point.source)
                self._publish(error="I couldn't save that choice. Please try again.")
        elif (
            kind == ActionKind.NAVIGATE
            and point is not None
            and choice.target is not None
        ):
            self.persistent = True
            self.task = asyncio.create_task(
                self._navigate(point, choice.target, self.identifier)
            )

    def _remove(self, point: Point) -> None:
        self.points = tuple(
            item
            for item in self.points
            if not set(item.feedback_sources).intersection(point.feedback_sources)
        )
        self.selected = None
        if self.points:
            self.mode = "rounds"
            self._publish()
        else:
            self.clear()

    async def _navigate(self, point: Point, target: Target, token: str) -> None:
        result = await self.navigator.open(target)
        if token != self.identifier:
            return
        if not result.success:
            log.info(
                "interaction.navigation_failed", target=target.value, error=result.error
            )
            self._publish(error="That place isn't available. Try another choice.")
            return
        try:
            self.memory.record_source_feedback(
                point.feedback_sources, "engaged", self.memory.clock()
            )
            self.attention.restore_sources(self.memory.load_source_attention())
        except sqlite3.Error:
            log.exception("interaction.engagement_failed", source=point.source)
        self.clear(cancel_task=False)

    def clear(self, *, cancel_task: bool = True) -> None:
        if cancel_task and self.task is not None:
            self.task.cancel()
        self.delivery.clear()
        self.identifier = ""
        self.choices.clear()
        self.points = ()
        self.selected = None
        self.engaged = False
        self.persistent = False
        if cancel_task:
            self.dismissed()
        self.emit({"dismiss_conversation": True})

    async def close(self) -> None:
        task = self.task
        self.clear()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
