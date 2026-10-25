from dataclasses import dataclass
from enum import StrEnum

from pydantic import TypeAdapter

from ..domain import ExecutionOutcome
from ..identity import SessionIdentity
from ..types import JSONObject


class SourceKind(StrEnum):
    REQUEST = "request"
    PROGRESS = "reported_progress"
    TURN_ENDED = "turn_ended"


@dataclass(frozen=True)
class NarrativeSource:
    identifier: str
    kind: SourceKind
    text: str
    scope: str
    outcome: ExecutionOutcome | None
    origin: SessionIdentity | None
    observed_at: float | None
    full_text: str
    scope_revision: str | None = None

    @classmethod
    def from_record(cls, identifier: str, record: JSONObject) -> NarrativeSource:
        return SOURCE_ADAPTER.validate_python(
            {
                "identifier": identifier,
                "kind": record["kind"],
                "text": record.get("text", ""),
                "scope": record["scope"],
                "outcome": record.get("outcome"),
                "origin": record.get("origin"),
                "observed_at": record.get("observed_at"),
                "full_text": record.get("text", ""),
                "scope_revision": record.get("scope_revision"),
            }
        )

    def record(self) -> JSONObject:
        result: JSONObject = {"kind": self.kind.value, "scope": self.scope}
        if self.text:
            result["text"] = self.text
        if self.outcome is not None:
            result["outcome"] = self.outcome.value
        if self.observed_at is not None:
            result["observed_at"] = self.observed_at
        if self.origin is not None:
            result["origin"] = {
                "source": self.origin.source.value,
                "native_id": self.origin.native_id,
            }
        if self.scope_revision is not None:
            result["scope_revision"] = self.scope_revision
        return result


SOURCE_ADAPTER = TypeAdapter(NarrativeSource)
