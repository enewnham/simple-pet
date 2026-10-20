import json
import re
from urllib.parse import urlsplit


KINDS = {"branch", "pull_request", "document", "ticket", "patch"}
FAILURES = {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE"}
TABLES = ("work_items", "work_sessions", "work_memberships", "work_aliases", "work_artifacts", "work_links",
          "work_evidence", "work_opportunities", "work_surfacings")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def repository_identity(value):
    if not value or value.startswith("local:"):
        return value
    if "://" in value:
        parsed = urlsplit(value)
        value = (parsed.hostname or "") + parsed.path
    else:
        value = re.sub(r"^(?:[^/@]+@)?([^/:]+):(.+)$", r"\1/\2", value)
    return value.removesuffix("/").removesuffix(".git").lower()


def text(value, name, *, optional=False):
    if value is None and optional:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def descriptor(value):
    if not isinstance(value, dict):
        raise ValueError("artifact must be an object")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("unsupported artifact kind")
    result = {"kind": kind, "ref": text(value.get("ref"), "artifact.ref")}
    for key in ("repository", "revision", "checkout", "title"):
        if value.get(key) not in (None, ""):
            result[key] = text(value[key], f"artifact.{key}")
    if "\0" in result.get("checkout", ""):
        raise ValueError("artifact.checkout must be a filesystem path without NUL characters")
    result["repository"] = repository_identity(result.get("repository", ""))
    if kind == "branch" and not result["repository"]:
        raise ValueError("branch artifacts require repository")
    if kind == "pull_request" and not re.match(r"https?://[^/]+/[^/]+/[^/]+/pull/\d+/?$", result["ref"]):
        raise ValueError("pull_request.ref must be a pull request URL")
    if kind == "pull_request":
        result["ref"] = result["ref"].rstrip("/")
        parsed = urlsplit(result["ref"])
        result["repository"] = repository_identity(parsed.netloc + parsed.path.rsplit("/pull/", 1)[0])
    return result
