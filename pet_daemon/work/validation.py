import json
import re
from urllib.parse import urlsplit

from ..domain import ArtifactKind
from ..repository_identity import parse_remote_repository
from .models import ArtifactReference


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda value: value.model_dump(mode="json"),
    )


def text(value: object, name: str, *, optional: bool = False) -> str:
    if value is None and optional:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def normalize_reference(value: ArtifactReference) -> ArtifactReference:
    result = value.model_copy()
    result.kind = ArtifactKind(value.kind)
    result.ref = text(value.ref, "artifact.ref")
    result.revision = (
        text(value.revision, "artifact.revision")
        if value.revision is not None
        else None
    )
    result.checkout = (
        text(value.checkout, "artifact.checkout")
        if value.checkout is not None
        else None
    )
    result.title = (
        text(value.title, "artifact.title") if value.title is not None else None
    )
    if result.checkout is not None and "\0" in result.checkout:
        raise ValueError(
            "artifact.checkout must be a filesystem path without NUL characters"
        )
    if result.kind == ArtifactKind.BRANCH and not result.repository:
        raise ValueError("branch artifacts require repository")
    if result.kind == ArtifactKind.PULL_REQUEST and not re.match(
        r"https?://[^/]+/[^/]+/[^/]+/pull/\d+/?$", result.ref
    ):
        raise ValueError("pull_request.ref must be a pull request URL")
    if result.kind == ArtifactKind.PULL_REQUEST:
        ref = result.ref.rstrip("/")
        parsed = urlsplit(ref)
        result.ref = ref
        result.repository = parse_remote_repository(
            parsed.netloc + parsed.path.rsplit("/pull/", 1)[0]
        )
    return result
