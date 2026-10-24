from dataclasses import dataclass
from typing import TypedDict

from .domain import SessionSource


@dataclass(frozen=True, order=True)
class SessionIdentity:
    source: SessionSource
    native_id: str


class SessionData(TypedDict):
    source: str
    native_id: str


def session_data(identity: SessionIdentity) -> SessionData:
    return {"source": identity.source.value, "native_id": identity.native_id}


def session_from_data(data: SessionData) -> SessionIdentity:
    return SessionIdentity(SessionSource(data["source"]), data["native_id"])
