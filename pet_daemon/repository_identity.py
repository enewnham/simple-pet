import re
from typing import Annotated, Literal, overload
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field


class RemoteRepository(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str
    kind: Literal["remote"] = "remote"

    def __lt__(self, other: RemoteRepository) -> bool:
        return self.value < other.value


class LocalRepository(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: str
    kind: Literal["local"] = "local"

    def __lt__(self, other: LocalRepository) -> bool:
        return self.path < other.path


type RepositoryIdentity = Annotated[
    RemoteRepository | LocalRepository, Field(discriminator="kind")
]


def parse_remote_repository(value: str) -> RemoteRepository | None:
    value = value.strip().rstrip("/")
    if "://" in value:
        parsed = urlsplit(value)
        if not parsed.hostname:
            return None
        host, path = parsed.hostname, parsed.path
    else:
        match = re.fullmatch(r"(?:[^@/]+@)?([^/:]+):(.+)", value)
        if match:
            host, path = match.groups()
        elif "/" in value and not value.startswith(("/", ".")):
            host, path = value.split("/", 1)
        else:
            return None
    path = path.strip("/").removesuffix(".git")
    return RemoteRepository(value=f"{host}/{path}".lower()) if path else None


@overload
def repository_columns(value: RepositoryIdentity) -> tuple[str, str]: ...


@overload
def repository_columns(value: None) -> tuple[None, None]: ...


def repository_columns(
    value: RepositoryIdentity | None,
) -> tuple[str | None, str | None]:
    match value:
        case RemoteRepository():
            return "remote", value.value
        case LocalRepository():
            return "local", value.path
        case None:
            return None, None


def repository_from_columns(
    kind: str | None, value: str | None
) -> RepositoryIdentity | None:
    if kind is None:
        return None
    if value is None:
        raise ValueError("Repository identity requires a value")
    if kind == "remote":
        return RemoteRepository(value=value)
    if kind == "local":
        return LocalRepository(path=value)
    raise ValueError(f"Unsupported repository kind: {kind}")
