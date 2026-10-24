from collections.abc import Mapping

from pydantic import ValidationError

from ..domain import ExecutionOutcome
from ..runtime import HookEvent, ProcessIdentity


def normalize_hook(packet: object) -> HookEvent | None:
    if not isinstance(packet, Mapping):
        return None
    nullable = {"manual", "owner", "exit_code"}
    if any(
        value is None and name in HookEvent.model_fields and name not in nullable
        for name, value in packet.items()
    ):
        return None
    if "aliases" in packet and not isinstance(packet["aliases"], list):
        return None
    data = dict(packet)
    data.pop("kind", None)
    if "manual" in data:
        data["manual"] = bool(data["manual"])
    data["owner"] = ProcessIdentity.parse(data.get("owner"))
    if type(data.get("exit_code")) is not int:
        data.pop("exit_code", None)
    try:
        event = HookEvent.model_validate(data)
    except ValidationError:
        return None
    if event.outcome == ExecutionOutcome.STALE or (event.repo and "\0" in event.repo):
        return None
    return event
