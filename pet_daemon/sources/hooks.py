from dataclasses import dataclass, fields
from collections.abc import Mapping

from ..runtime import ProcessIdentity


EVENTS = frozenset({
    "session_start", "session_end", "intent", "tool_start", "tool_done",
    "tool_error", "turn_done", "turn_error", "interrupt", "subagent_start",
    "subagent_done", "progress", "waiting",
})
HARNESS = frozenset({"cli", "claude", "codex", "cursor"})
TEXT_FIELDS = ("parent", "parent_call", "call", "turn", "tool", "goal", "progress", "repo", "tool_name", "sentiment_text")


@dataclass(frozen=True)
class HookEvent:
    harness: str
    session: str
    event: str
    aliases: tuple[str, ...] = ()
    parent: str | None = None
    parent_call: str | None = None
    call: str | None = None
    turn: str | None = None
    tool: str | None = None
    goal: str | None = None
    progress: str | None = None
    repo: str | None = None
    tool_name: str | None = None
    sentiment_text: str | None = None
    outcome: str | None = None
    manual: bool | None = None
    owner: ProcessIdentity | None = None
    exit_code: int | None = None

    @property
    def source(self):
        return f"hook:{self.harness}"

    @property
    def subject(self):
        return f"{self.harness}:{self.session}"

    def packet(self):
        packet = {"kind": "event", "harness": self.harness, "session": self.session, "event": self.event}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in packet or value is None:
                continue
            if item.name == "aliases":
                if value:
                    packet[item.name] = list(value)
            elif item.name == "owner":
                packet[item.name] = value.packet()
            else:
                packet[item.name] = value
        return packet


def normalize_hook(packet):
    if not isinstance(packet, Mapping):
        return None
    harness, event, session = packet.get("harness"), packet.get("event"), packet.get("session")
    if (not isinstance(harness, str) or harness not in HARNESS
            or not isinstance(event, str) or event not in EVENTS
            or not isinstance(session, str) or not session):
        return None
    if any(name in packet and not isinstance(packet[name], str) for name in TEXT_FIELDS):
        return None
    aliases = packet.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(alias, str) and alias for alias in aliases):
        return None
    if "outcome" in packet and packet["outcome"] not in ("success", "error", "cancelled"):
        return None
    if "\0" in packet.get("repo", ""):
        return None
    values = {name: packet[name] for name in TEXT_FIELDS if name in packet}
    values.update(aliases=tuple(aliases), owner=ProcessIdentity.parse(packet.get("owner")))
    if "manual" in packet:
        values["manual"] = bool(packet["manual"])
    if "outcome" in packet:
        values["outcome"] = packet["outcome"]
    if type(packet.get("exit_code")) is int:
        values["exit_code"] = packet["exit_code"]
    return HookEvent(harness, session, event, **values)
