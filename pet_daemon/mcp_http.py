import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import uvicorn
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from pydantic import AnyHttpUrl
from starlette.applications import Starlette

if TYPE_CHECKING:
    from .mcp import WorkContext

DEFAULT_TOKEN_FILE = Path(__file__).resolve().parents[1] / "state/pet-mcp.token"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def create_token_file(path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            os.fchmod(stream.fileno(), 0o600)
            stream.write((secrets.token_urlsafe(32) + "\n").encode("ascii"))
            stream.flush()
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()


def load_token(path: Path) -> str:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        create_token_file(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError(
                "MCP token file must be an owned regular file with mode 0600"
            )
        with os.fdopen(descriptor, encoding="ascii", closefd=False) as stream:
            try:
                token = stream.read(4097).strip()
            except UnicodeError:
                raise ValueError(
                    "MCP token file must contain an ASCII bearer token"
                ) from None
        if (
            not 32 <= len(token) <= 4096
            or re.fullmatch(r"[-A-Za-z0-9._~+/]+=*", token) is None
        ):
            raise ValueError(
                "MCP token file must contain a bearer token of 32–4096 characters"
            )
        return token
    finally:
        os.close(descriptor)


class SharedToken:
    def __init__(self, token: str, write_token: str | None = None) -> None:
        self.token = token.encode()
        self.write_token = write_token.encode() if write_token else None
        if self.write_token is not None and secrets.compare_digest(
            self.token, self.write_token
        ):
            raise ValueError("Read and write credentials must be distinct")

    async def verify_token(self, token: str) -> AccessToken | None:
        from .mcp import WRITE_PERMISSION

        value = token.encode()
        if secrets.compare_digest(value, self.token):
            return AccessToken(token=token, client_id="pet-mcp", scopes=[])
        if self.write_token is not None and secrets.compare_digest(
            value, self.write_token
        ):
            return AccessToken(
                token=token, client_id="pet-mcp-writer", scopes=[WRITE_PERMISSION]
            )
        return None


def http_application(
    context: WorkContext,
    token: str,
    host: str,
    port: int,
    write_token: str | None = None,
) -> Starlette:
    from .mcp import create_server

    if host not in LOOPBACK_HOSTS:
        raise ValueError("MCP HTTP must bind to a loopback host")
    address = f"[{host}]" if host == "::1" else host
    auth = AuthSettings(
        issuer_url=AnyHttpUrl(f"http://{address}:{port}"), resource_server_url=None
    )
    server = create_server(
        context,
        token_verifier=SharedToken(token, write_token),
        auth=auth,
    )
    return server.streamable_http_app(
        host=host, stateless_http=True, json_response=True
    )


def run_http(
    context: WorkContext,
    token_file: Path,
    host: str,
    port: int,
    write_token_file: Path | None = None,
) -> None:
    token = load_token(token_file)
    write_token = load_token(write_token_file) if write_token_file is not None else None
    application = http_application(context, token, host, port, write_token)
    uvicorn.run(
        application,
        host=host,
        port=port,
        access_log=False,
        log_level="info",
        proxy_headers=False,
    )
