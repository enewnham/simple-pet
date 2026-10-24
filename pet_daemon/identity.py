from pydantic import BaseModel, ConfigDict

from .domain import SessionSource


class SessionIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)
    source: SessionSource
    native_id: str

    def __lt__(self, other: SessionIdentity) -> bool:
        return (self.source, self.native_id) < (other.source, other.native_id)


def session_data(identity: SessionIdentity) -> dict[str, object]:
    return identity.model_dump(mode="json")
