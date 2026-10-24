import sqlite3

from ..domain import SessionSource
from ..identity import SessionIdentity


def session_values(identity: SessionIdentity) -> tuple[str, str]:
    return identity.source.value, identity.native_id


def session_from_row(row: sqlite3.Row, prefix: str = "") -> SessionIdentity:
    return SessionIdentity(
        SessionSource(row[f"{prefix}source"]), row[f"{prefix}session"]
    )


def optional_session_values(
    identity: SessionIdentity | None,
) -> tuple[str | None, str | None]:
    return session_values(identity) if identity is not None else (None, None)


def optional_session_from_row(row: sqlite3.Row, prefix: str) -> SessionIdentity | None:
    return session_from_row(row, prefix) if row[f"{prefix}source"] is not None else None


def session_parameters(
    identities: tuple[SessionIdentity, ...],
) -> tuple[str, tuple[str, ...]]:
    placeholders = ",".join("(?,?)" for _ in identities)
    values = tuple(
        value for identity in identities for value in session_values(identity)
    )
    return placeholders, values
