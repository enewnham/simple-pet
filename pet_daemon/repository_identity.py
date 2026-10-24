import re
from dataclasses import dataclass
from typing import Literal, TypedDict, overload
from urllib.parse import urlsplit


@dataclass(frozen=True, order=True)
class RemoteRepository:
    value: str


@dataclass(frozen=True, order=True)
class LocalRepository:
    path: str


type RepositoryIdentity = RemoteRepository | LocalRepository


class RemoteRepositoryData(TypedDict):
    kind: Literal["remote"]
    value: str


class LocalRepositoryData(TypedDict):
    kind: Literal["local"]
    path: str


type RepositoryData = RemoteRepositoryData | LocalRepositoryData


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
    return RemoteRepository(f"{host}/{path}".lower()) if path else None


@overload
def repository_data(value: RepositoryIdentity) -> RepositoryData: ...


@overload
def repository_data(value: None) -> None: ...


def repository_data(value: RepositoryIdentity | None) -> RepositoryData | None:
    match value:
        case RemoteRepository():
            return {"kind": "remote", "value": value.value}
        case LocalRepository():
            return {"kind": "local", "path": value.path}
        case None:
            return None


@overload
def repository_from_data(value: RepositoryData) -> RepositoryIdentity: ...


@overload
def repository_from_data(value: None) -> None: ...


def repository_from_data(value: RepositoryData | None) -> RepositoryIdentity | None:
    if value is None:
        return None
    if value["kind"] == "remote":
        return RemoteRepository(value["value"])
    return LocalRepository(value["path"])


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
        return RemoteRepository(value)
    if kind == "local":
        return LocalRepository(value)
    raise ValueError(f"Unsupported repository kind: {kind}")


def remote_repository_from_data(value: RepositoryData | None) -> RemoteRepository:
    if value is None or value["kind"] != "remote":
        raise ValueError("Pull request requires a remote repository")
    return RemoteRepository(value["value"])
