import json
import re
from dataclasses import replace
from urllib.parse import urlsplit

from ..domain import ArtifactKind
from .models import ArtifactReference


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def repository_identity(value: str) -> str:
    if not value or value.startswith("local:"):
        return value
    if "://" in value:
        parsed = urlsplit(value)
        value = (parsed.hostname or "") + parsed.path
    else:
        value = re.sub(r"^(?:[^/@]+@)?([^/:]+):(.+)$", r"\1/\2", value)
    return value.removesuffix("/").removesuffix(".git").lower()


def text(value: object, name: str, *, optional: bool = False) -> str:
    if value is None and optional:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def normalize_reference(value: ArtifactReference) -> ArtifactReference:
    result = replace(
        value,
        kind=ArtifactKind(value.kind),
        ref=text(value.ref, "artifact.ref"),
        repository=text(value.repository, "artifact.repository")
        if value.repository
        else "",
        revision=text(value.revision, "artifact.revision")
        if value.revision is not None
        else None,
        checkout=text(value.checkout, "artifact.checkout")
        if value.checkout is not None
        else None,
        title=text(value.title, "artifact.title") if value.title is not None else None,
    )
    if result.checkout is not None and "\0" in result.checkout:
        raise ValueError(
            "artifact.checkout must be a filesystem path without NUL characters"
        )
    result = replace(result, repository=repository_identity(result.repository))
    if result.kind == ArtifactKind.BRANCH and not result.repository:
        raise ValueError("branch artifacts require repository")
    if result.kind == ArtifactKind.PULL_REQUEST and not re.match(
        r"https?://[^/]+/[^/]+/[^/]+/pull/\d+/?$", result.ref
    ):
        raise ValueError("pull_request.ref must be a pull request URL")
    if result.kind == ArtifactKind.PULL_REQUEST:
        ref = result.ref.rstrip("/")
        parsed = urlsplit(ref)
        result = replace(
            result,
            ref=ref,
            repository=repository_identity(
                parsed.netloc + parsed.path.rsplit("/pull/", 1)[0]
            ),
        )
    return result
