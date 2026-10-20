from copy import deepcopy
from dataclasses import dataclass, replace
import json
from collections.abc import Mapping

from .work import Opportunity, WorkSubject

from .telemetry import trace


@dataclass(frozen=True)
class AttentionRequest:
    reason: str
    context: str | None = None
    ttl: float | None = None
    trigger: dict | None = None
    eligible: frozenset[str] | None = None


@dataclass(frozen=True)
class SelectionSupport:
    subject_revision: str | None = None
    action_version: int | None = None
    artifact_ref: str | None = None
    artifact_revision: str | None = None
    session_transition: tuple = ()

    @classmethod
    def capture(cls, subject, action, focus):
        return cls(
            subject.revision if subject else None,
            action.version if action else None,
            action.artifact.get("ref") if action else None,
            action.artifact.get("revision") if action else None,
            tuple(
                focus.get(key)
                for key in ("generation", "turn_active", "waiting_for_input", "outcome")
            )
            if focus
            else (),
        )


@dataclass(frozen=True)
class AttentionSelection:
    subject: WorkSubject | None
    focus_session: Mapping | None
    context_sessions: tuple[Mapping, ...]
    reason: str
    situation: str
    context: str | None = None
    action: Opportunity | None = None
    supporting_facts: SelectionSupport = SelectionSupport()

    @property
    def id(self):
        return (
            self.subject.id
            if self.subject is not None
            else self.focus_session["id"]
            if self.focus_session
            else None
        )

    def supported(self, snapshot):
        current = self.refresh(snapshot)
        if self.subject is not None and current.subject is None:
            return False
        if self.action is not None:
            if current.action is None or current.action.kind != self.action.kind:
                return False
            before, after = self.supporting_facts, current.supporting_facts
            if (before.artifact_ref, before.artifact_revision) != (
                after.artifact_ref,
                after.artifact_revision,
            ):
                return False
        if self.focus_session is not None and self.action is None:
            if current.focus_session is None:
                return False
            if (
                self.supporting_facts.session_transition
                != current.supporting_facts.session_transition
            ):
                return False
        return True

    def refresh(self, snapshot):
        subjects = {subject.id: subject for subject in snapshot.get("work", ())}
        sessions = {session["id"]: session for session in snapshot.get("sessions", ())}
        subject = subjects.get(self.subject.id) if self.subject is not None else None
        action = self.action
        if action is not None:
            action = next(
                (
                    item
                    for item in (subject.view.opportunities if subject else ())
                    if item.id == self.action.id
                ),
                None,
            )
        focus = sessions.get(self.focus_session["id"]) if self.focus_session else None
        return replace(
            self,
            subject=subject,
            action=deepcopy(action),
            supporting_facts=SelectionSupport.capture(subject, action, focus),
            focus_session=deepcopy(focus),
            context_sessions=tuple(
                deepcopy(sessions[item["id"]])
                for item in self.context_sessions
                if item["id"] in sessions
            ),
        )


def _sessions(subject, snapshot):
    identifiers = set(subject.sessions)
    if subject.kind == "session":
        identifiers.add(subject.id)
    return [
        session
        for session in snapshot.get("sessions", ())
        if session["id"] in identifiers
    ]


def _session_tier(session):
    if session.get("waiting_for_input"):
        return 0
    if session.get("current_obstacle"):
        return 1
    if session.get("turn_active") and not session.get("activity_uncertain"):
        return 3
    if session.get("outcome") and session.get("quiet_seconds", 0) < 300:
        return 4
    return 5


def _action_tier(action):
    return 1 if action.kind == "resolve_checks" else 2


class Attention:
    def __init__(self, config, clock, rng, history=None):
        self.config = config
        self.clock = clock
        self.rng = rng
        self.pending = None
        self.last_attempt = float("-inf")
        self.next_observation = clock() + rng.uniform(*config.observation_seconds)
        self.selections = {}
        self.considered = {}
        self.sequence = 0
        if history:
            self.restore(history)

    def reset(self):
        self.pending = None
        self.last_attempt = float("-inf")
        self.next_observation = self.clock() + self.rng.uniform(
            *self.config.observation_seconds
        )
        self.selections.clear()
        self.considered.clear()
        self.sequence = 0

    def queue(self, reason, trigger=None, failed=False):
        now = self.clock()
        if reason == "failure":
            self.pending = (reason, now + 1, trigger)
        elif (
            reason == "check_in"
            or not self.pending
            or self.pending[0] != "failure"
            or not failed
        ):
            delay = {"prompt": 0, "completion": 1, "check_in": 0}[reason]
            self.pending = (reason, now + delay, trigger)
        trace(
            "observation.queued",
            reason=self.pending[0],
            scheduled_at=self.pending[1],
            session=(self.pending[2] or {}).get("target_session"),
            scheduled=self.pending,
        )

    def routine_eligible(self, subject, snapshot):
        if not self.is_new(subject):
            return False
        view = subject.view
        if (
            view.intent
            or view.requests
            or view.activity
            or view.opportunities
            or view.current_obstacle
        ):
            return True
        return any(
            session.get("turn_active")
            or session.get("waiting_for_input")
            or session.get("outcome")
            for session in _sessions(subject, snapshot)
        )

    def select(self, snapshot, reason="routine", context=None, eligible=None):
        trigger = snapshot.get("trigger") or {}
        subject = None
        action = None
        focus = None
        others = ()
        if context is None and reason not in ("explicit", "care"):
            target_session = trigger.get("target_session") or (
                f"{trigger['harness']}:{trigger['session']}"
                if trigger.get("harness") and trigger.get("session")
                else None
            )
            if reason == "click":
                target_session = None
            target_subject = trigger.get("target_subject") or trigger.get(
                "target_opportunity"
            )
            subjects = [
                subject
                for subject in snapshot.get("work", ())
                if reason != "routine" or self.routine_eligible(subject, snapshot)
            ]
            candidates = []
            for item in subjects:
                sessions = _sessions(item, snapshot)
                if eligible is not None:
                    sessions = [
                        session for session in sessions if session["id"] in eligible
                    ]
                    if not sessions and item.id not in eligible:
                        continue
                actions = item.view.opportunities
                tiers = [_session_tier(session) for session in sessions] + [
                    _action_tier(value) for value in actions
                ]
                tier = min((*tiers, 1 if item.view.current_obstacle else 5))
                candidates.append((item, sessions, actions, tier))
            targeted = [
                item
                for item in candidates
                if item[0].id == target_subject
                or any(value.id == target_subject for value in item[2])
                or target_session
                and any(session["id"] == target_session for session in item[1])
            ]
            if targeted:
                candidates = targeted
            elif target_session:
                candidates = []
            if candidates:
                subject, sessions, actions, tier = min(
                    candidates,
                    key=lambda item: (
                        item[3],
                        self.selections.get(item[0].id, -1),
                        -max(
                            (
                                session.get("waiting_seconds", 0)
                                for session in item[1]
                                if session.get("waiting_for_input")
                            ),
                            default=0,
                        )
                        if reason == "click" and item[3] == 0
                        else -item[0].changed_at,
                        item[0].id,
                    ),
                )
                if target_session:
                    focus = next(
                        (
                            session
                            for session in sessions
                            if session["id"] == target_session
                        ),
                        None,
                    )
                if focus is None and sessions:
                    focus = min(
                        sessions,
                        key=lambda session: (
                            _session_tier(session),
                            self.selections.get(session["id"], -1),
                            -session.get("waiting_seconds", 0)
                            if session.get("waiting_for_input")
                            else session.get("quiet_seconds", 0)
                            if not session.get("turn_active")
                            else -session.get("quiet_seconds", 0),
                            session["id"],
                        ),
                    )
                if not target_session and reason in ("routine", "click"):
                    action = next(
                        (value for value in actions if value.id == target_subject), None
                    )
                    if action is None:
                        action = min(
                            actions,
                            key=lambda value: (
                                _action_tier(value),
                                -value.changed_at,
                                value.id,
                            ),
                            default=None,
                        )
                    if (
                        action is not None
                        and focus is not None
                        and _session_tier(focus) < _action_tier(action)
                    ):
                        action = None
                others = tuple(
                    sorted(
                        (
                            session
                            for session in sessions
                            if focus is None or session["id"] != focus["id"]
                        ),
                        key=lambda session: (_session_tier(session), session["id"]),
                    )
                )
            else:
                sessions = [
                    session
                    for session in snapshot.get("sessions", ())
                    if eligible is None or session["id"] in eligible
                ]
                preferred = (
                    [session for session in sessions if session["id"] == target_session]
                    if target_session
                    else sessions
                )
                if preferred:
                    focus = min(
                        preferred,
                        key=lambda session: (
                            _session_tier(session),
                            self.selections.get(session["id"], -1),
                            -session.get("waiting_seconds", 0)
                            if session.get("waiting_for_input")
                            else session.get("quiet_seconds", 0)
                            if not session.get("turn_active")
                            else -session.get("quiet_seconds", 0),
                            session["id"],
                        ),
                    )
                    others = tuple(
                        sorted(
                            (
                                session
                                for session in sessions
                                if session["id"] != focus["id"]
                            ),
                            key=lambda session: (_session_tier(session), session["id"]),
                        )
                    )
            if (
                reason == "click"
                and action is None
                and not any(
                    session.get("waiting_for_input")
                    or (
                        session.get("turn_active")
                        and not session.get("activity_uncertain")
                    )
                    for session in ((focus,) if focus else ())
                )
            ):
                subject, focus, others = None, None, ()
        situation = reason
        if context is not None:
            situation = "explicit"
        elif reason == "click":
            situation = (
                "click_waiting"
                if focus and focus.get("waiting_for_input")
                else "click_working"
                if focus
                else "click_idle"
            )
        elif reason == "care":
            situation = "care_" + trigger["care"]
        elif reason == "check_in":
            situation = (
                "check_in_waiting"
                if focus and focus.get("waiting_for_input")
                else "check_in_running"
                if focus and focus.get("active_tools")
                else "check_in_quiet"
            )
        if action is not None:
            situation = "opportunity"
        return AttentionSelection(
            subject,
            deepcopy(focus),
            deepcopy(others),
            reason,
            situation,
            context,
            deepcopy(action),
            SelectionSupport.capture(subject, action, focus),
        )

    def request(
        self,
        snapshot,
        *,
        check_in=(),
        manual=None,
        clicked=False,
        care=None,
        episode_relevant=lambda trigger: True,
    ):
        now = self.clock()
        if manual is not None:
            return AttentionRequest("explicit", manual[0], manual[1])
        if clicked:
            return AttentionRequest("click")
        if snapshot.get("sleeping", False):
            return None
        if self.pending and now >= self.pending[1]:
            reason, _, trigger = self.pending
            self.pending = None
            if not trigger or "episode" not in trigger or episode_relevant(trigger):
                return AttentionRequest(reason, trigger=trigger)
        if check_in:
            return AttentionRequest("check_in", eligible=frozenset(check_in))
        if now >= self.next_observation:
            self.next_observation = now + self.rng.uniform(
                *self.config.observation_seconds
            )
            if any(
                self.routine_eligible(subject, snapshot)
                for subject in snapshot.get("work", ())
            ):
                return AttentionRequest("routine")
        if care is not None and not snapshot.get(
            "engaged",
            any(session.get("turn_active") for session in snapshot.get("sessions", ())),
        ):
            return AttentionRequest("care", trigger={"care": care})
        return None

    def restore(self, history):
        self.selections = {
            identifier: row["last_selected"] for identifier, row in history.items()
        }
        self.considered = {
            identifier: row["considered_revision"]
            for identifier, row in history.items()
        }
        self.sequence = max(self.selections.values(), default=0)

    def is_new(self, subject):
        revision = json.dumps(subject.revision, sort_keys=True, separators=(",", ":"))
        return self.considered.get(subject.id) != revision

    def mark_considered(self, subject):
        self.considered[subject.id] = json.dumps(
            subject.revision, sort_keys=True, separators=(",", ":")
        )

    def history(self, subject_id):
        return self.selections.get(subject_id, 0), self.considered.get(subject_id)

    def mark_selected(self, subject_id):
        self.sequence = max(self.clock(), self.sequence + 0.000001)
        self.selections[subject_id] = self.sequence
        trace("attention.selected", session=subject_id)

    def allowed(self, reason, trigger=None):
        if (
            reason in ("click", "explicit")
            or trigger
            and trigger.get("episode") == "waiting"
        ):
            return True
        return (
            self.clock() - self.last_attempt
            >= self.config.behavior.automatic_speech_cooldown_seconds
        )

    def scheduled(self, subject_id=None):
        self.last_attempt = self.clock()
        self.next_observation = self.last_attempt + self.rng.uniform(
            *self.config.observation_seconds
        )
        if subject_id is not None:
            self.mark_selected(subject_id)
