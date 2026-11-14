import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .mcp_http import load_token

SECRET_PLACEHOLDER = "__PET_TUNNEL_KEY__"
ROUTES_PLACEHOLDER = "  # __PET_TUNNEL_ROUTES__\n"
ROUTE_METHODS = ("POST", "GET", "DELETE")
ROUTE_PATH = re.compile(r"(/[A-Za-z0-9._~-]+)+")
INTERNAL_URL = re.compile(
    r"https://[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*\.internal"
)


@dataclass(frozen=True)
class Route:
    """Forward one exact public path to an ngrok internal endpoint run by another local agent."""

    path: str
    forward: str

    def __post_init__(self) -> None:
        if ROUTE_PATH.fullmatch(self.path) is None or self.path == "/mcp":
            raise ValueError(
                f"Route path {self.path!r} must be an absolute path other than /mcp"
            )
        if INTERNAL_URL.fullmatch(self.forward) is None:
            raise ValueError(
                f"Route {self.path} must forward to an https://<name>.internal endpoint"
            )

    def rule(self) -> str:
        methods = ", ".join(f"'{method}'" for method in ROUTE_METHODS)
        rule = {
            "expressions": [
                f"req.url.path == '{self.path}' && req.method in [{methods}]"
            ],
            "actions": [{"type": "forward-internal", "config": {"url": self.forward}}],
        }
        return f"  - {json.dumps(rule)}\n"


def load_routes(path: Path) -> list[Route]:
    if not path.exists():
        return []
    entries = json.loads(path.read_text())
    routes = [Route(entry["path"], entry["forward"]) for entry in entries]
    if len({route.path for route in routes}) != len(routes):
        raise ValueError("Tunnel routes must use distinct paths")
    return routes


def render_policy(
    template: str, secret: str, routes: list[Route] | tuple[Route, ...] = ()
) -> str:
    if template.count(SECRET_PLACEHOLDER) != 1:
        raise ValueError("Tunnel policy template requires one secret placeholder")
    if template.count(ROUTES_PLACEHOLDER) != 1:
        raise ValueError("Tunnel policy template requires one routes placeholder")
    policy = template.replace(
        ROUTES_PLACEHOLDER, "".join(route.rule() for route in routes)
    )
    return policy.replace(SECRET_PLACEHOLDER, json.dumps(secret, ensure_ascii=True))


def write_policy(path: Path, policy: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(policy)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_policy(root: Path) -> Path:
    secret = load_token(root / "state/pet-mcp-tunnel.secret")
    template = (root / "examples/pet-mcp-ngrok.yml").read_text()
    routes = load_routes(root / "state/pet-tunnel-routes.json")
    policy = render_policy(template, secret, routes)
    path = root / "state/pet-mcp-ngrok.yml"
    write_policy(path, policy)
    return path
