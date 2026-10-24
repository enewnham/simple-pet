"""Small evidence helpers for investigating local pet behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .identity import SessionIdentity, session_data


@dataclass(frozen=True)
class LogEntry:
    source: Path
    line: int
    data: dict[str, Any]


@dataclass(frozen=True)
class LogDiagnostic:
    source: Path
    line: int
    error: str
    raw: str


@dataclass(frozen=True)
class LogEvidence:
    entries: tuple[LogEntry, ...]
    diagnostics: tuple[LogDiagnostic, ...]


@dataclass(frozen=True)
class DeliveryTrace:
    message_id: str
    anchors: tuple[LogEntry, ...]
    chain: tuple[LogEntry, ...]
    observations: tuple[tuple[str, str], ...]
    diagnostics: tuple[str, ...]


def load_logs(path: Path | str) -> LogEvidence:
    """Read a JSONL log and its numbered rotations from oldest to newest."""
    path = Path(path)
    rotations = sorted(
        (
            candidate
            for candidate in path.parent.glob(f"{path.name}.[0-9]*")
            if candidate.name.removeprefix(f"{path.name}.").isdigit()
        ),
        key=lambda candidate: int(candidate.name.removeprefix(f"{path.name}.")),
        reverse=True,
    )
    sources = [*rotations, path] if path.exists() else rotations
    if not sources:
        raise FileNotFoundError(path)
    entries: list[LogEntry] = []
    diagnostics: list[LogDiagnostic] = []
    for source in sources:
        with source.open("rb") as stream:
            for number, raw in enumerate(stream, 1):
                try:
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise TypeError("JSON record is not an object")
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    TypeError,
                ) as error:
                    diagnostics.append(
                        LogDiagnostic(
                            source, number, str(error), raw.decode("utf-8", "replace")
                        )
                    )
                    continue
                entries.append(LogEntry(source, number, value))
    return LogEvidence(tuple(entries), tuple(diagnostics))


def _timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Include a timezone offset when selecting log timestamps")
    return parsed


def select_logs(
    entries: Sequence[LogEntry],
    *,
    run_id: str | None = None,
    observation_id: str | None = None,
    session: SessionIdentity | None = None,
    event: str | None = None,
    packet: Mapping[str, object] | None = None,
    start: str | datetime | None = None,
    end: str | datetime | None = None,
) -> tuple[LogEntry, ...]:
    """Select exact log fields and an inclusive timestamp window."""
    lower = _timestamp(start)
    upper = _timestamp(end)
    selected = []
    for entry in entries:
        data = entry.data
        if run_id is not None and data.get("run_id") != run_id:
            continue
        if observation_id is not None and data.get("observation_id") != observation_id:
            continue
        if session is not None and data.get("session") != session_data(session):
            continue
        if event is not None and data.get("event") != event:
            continue
        if packet is not None:
            payload = data.get("packet")
            if not isinstance(payload, dict) or any(
                payload.get(key) != value for key, value in packet.items()
            ):
                continue
        if lower is not None or upper is not None:
            stamp = data.get("timestamp")
            if not isinstance(stamp, str):
                continue
            try:
                observed = _timestamp(stamp)
                if (
                    observed is None
                    or (lower is not None and observed < lower)
                    or (upper is not None and observed > upper)
                ):
                    continue
            except ValueError, TypeError:
                continue
        selected.append(entry)
    return tuple(selected)


def trace_delivery(entries: Sequence[LogEntry], message_id: str) -> DeliveryTrace:
    """Join a delivery UUID to its request through exact structured fields."""
    anchors: list[LogEntry] = []
    observations: set[tuple[str, str]] = set()
    for entry in entries:
        data = entry.data
        packet = data.get("packet")
        matched = (
            data.get("event") == "speech.delivery" and data.get("message") == message_id
        ) or (
            data.get("event") == "renderer.command"
            and isinstance(packet, dict)
            and packet.get("delivery_id") == message_id
        )
        if not matched:
            continue
        anchors.append(entry)
        run_id = data.get("run_id")
        observation_id = data.get("observation_id")
        if isinstance(run_id, str) and isinstance(observation_id, str):
            observations.add((run_id, observation_id))

    anchor_locations = {(entry.source, entry.line) for entry in anchors}
    chain = tuple(
        entry
        for entry in entries
        if (entry.source, entry.line) in anchor_locations
        or (entry.data.get("run_id"), entry.data.get("observation_id")) in observations
    )
    diagnostics = []
    if not anchors:
        diagnostics.append("No structural delivery or renderer command record")
    if not any(entry.data.get("event") == "speech.delivery" for entry in anchors):
        diagnostics.append("No speech.delivery record")
    if not any(entry.data.get("event") == "renderer.command" for entry in anchors):
        diagnostics.append("No renderer.command record")
    if anchors and not observations:
        diagnostics.append("No matching run_id and observation_id pair")
    return DeliveryTrace(
        message_id,
        tuple(anchors),
        chain,
        tuple(sorted(observations)),
        tuple(diagnostics),
    )


def save_case(
    root: Path | str,
    name: str,
    *,
    report: str,
    case: Mapping[str, object],
    result: Mapping[str, object],
    script: str | None = None,
) -> Path:
    """Save the current case and a numbered investigation iteration."""
    if (
        not name
        or name in {".", ".."}
        or any(character in name for character in ("/", "\\"))
    ):
        raise ValueError("case name must be one path component")
    directory = Path(root) / name
    iterations = directory / "iterations"
    iterations.mkdir(parents=True, exist_ok=True)
    number = 1
    while (iterations / f"{number:04d}").exists():
        number += 1
    snapshot = iterations / f"{number:04d}"
    snapshot.mkdir()
    case_text = json.dumps(case, indent=2, ensure_ascii=False) + "\n"
    result_text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    (snapshot / "report.md").write_text(report, encoding="utf-8")
    (snapshot / "case.json").write_text(case_text, encoding="utf-8")
    (snapshot / "result.json").write_text(result_text, encoding="utf-8")
    if script is not None:
        (snapshot / "investigate.py").write_text(script, encoding="utf-8")
    (directory / "report.md").write_text(report, encoding="utf-8")
    (directory / "case.json").write_text(case_text, encoding="utf-8")
    return snapshot
