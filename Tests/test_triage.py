import json
import uuid
from pathlib import Path

import pytest

from pet_daemon.triage import load_logs, save_case, select_logs, trace_delivery


def write_jsonl(path: Path, *records: object) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_load_logs_reads_numbered_rotations_oldest_first(tmp_path: Path) -> None:
    log = tmp_path / "pet.jsonl"
    write_jsonl(tmp_path / "pet.jsonl.2", {"event": "first"})
    write_jsonl(tmp_path / "pet.jsonl.1", {"event": "second"}, "unexpected")
    write_jsonl(log, {"event": "third"})

    evidence = load_logs(log)

    assert [entry.data["event"] for entry in evidence.entries] == [
        "first",
        "second",
        "third",
    ]
    assert [entry.source.name for entry in evidence.entries] == [
        "pet.jsonl.2",
        "pet.jsonl.1",
        "pet.jsonl",
    ]
    assert evidence.diagnostics[0].source.name == "pet.jsonl.1"
    assert evidence.diagnostics[0].line == 2
    assert evidence.diagnostics[0].raw == '"unexpected"\n'


def test_trace_delivery_uses_packet_identifier_and_keeps_log_provenance(
    tmp_path: Path,
) -> None:
    message_id = str(uuid.uuid4())
    run_id = uuid.uuid4().hex
    observation_id = uuid.uuid4().hex
    log = tmp_path / "pet.jsonl"
    older = tmp_path / "pet.jsonl.1"
    write_jsonl(
        older,
        {
            "event": "hook.received",
            "run_id": run_id,
            "packet": {"goal": f"Find why {message_id} was spoken"},
        },
        {
            "event": "observation.started",
            "run_id": run_id,
            "observation_id": observation_id,
            "unrecognized": {"kept": True},
        },
    )
    write_jsonl(
        log,
        {
            "event": "renderer.command",
            "run_id": run_id,
            "observation_id": observation_id,
            "packet": {"text": "Hello", "delivery_id": message_id},
        },
        {"event": "speech.delivery", "run_id": run_id, "message": message_id},
        {
            "event": "renderer.command",
            "run_id": run_id,
            "observation_id": uuid.uuid4().hex,
            "packet": {"text": message_id, "delivery_id": str(uuid.uuid4())},
        },
    )

    evidence = load_logs(log)
    trace = trace_delivery(evidence.entries, message_id)

    assert trace.observations == ((run_id, observation_id),)
    assert [entry.data["event"] for entry in trace.anchors] == [
        "renderer.command",
        "speech.delivery",
    ]
    assert [entry.data["event"] for entry in trace.chain] == [
        "observation.started",
        "renderer.command",
        "speech.delivery",
    ]
    assert trace.chain[0].source == older
    assert trace.chain[0].line == 2
    assert trace.chain[0].data["unrecognized"] == {"kept": True}
    assert not trace.diagnostics


def test_incomplete_delivery_and_malformed_lines_remain_visible(tmp_path: Path) -> None:
    message_id = str(uuid.uuid4())
    log = tmp_path / "pet.jsonl"
    log.write_bytes(
        b'{"event":"hook.received","packet":{"goal":"'
        + message_id.encode()
        + b'"}}\n'
        + b'{"event":"speech.delivery","message":"'
        + message_id.encode()
        + b'","run_id":"run"}\n'
        + b'{"event":'
    )

    evidence = load_logs(log)
    trace = trace_delivery(evidence.entries, message_id)

    assert len(evidence.diagnostics) == 1
    assert evidence.diagnostics[0].line == 3
    assert evidence.diagnostics[0].raw == '{"event":'
    assert trace.observations == ()
    assert [entry.data["event"] for entry in trace.chain] == ["speech.delivery"]
    assert "No renderer.command record" in trace.diagnostics
    assert "No matching run_id and observation_id pair" in trace.diagnostics
    assert not trace_delivery(evidence.entries, str(uuid.uuid4())).anchors


def test_select_logs_filters_exact_fields_and_time(tmp_path: Path) -> None:
    log = tmp_path / "pet.jsonl"
    write_jsonl(
        log,
        {
            "event": "renderer.command",
            "run_id": "run-a",
            "observation_id": "observation-a",
            "session": "session-a",
            "timestamp": "2026-10-23T12:00:00Z",
            "packet": {"animation": "sleep"},
        },
        {
            "event": "renderer.command",
            "run_id": "run-a",
            "observation_id": "observation-b",
            "session": "session-a",
            "timestamp": "2026-10-23T12:02:00Z",
            "packet": {"animation": "wake"},
        },
        {
            "event": "renderer.command",
            "run_id": "run-b",
            "observation_id": "observation-a",
            "session": "session-a",
            "timestamp": "2026-10-23T12:00:00Z",
            "packet": {"animation": "sleep"},
        },
    )

    selected = select_logs(
        load_logs(log).entries,
        run_id="run-a",
        observation_id="observation-a",
        session="session-a",
        event="renderer.command",
        packet={"animation": "sleep"},
        start="2026-10-23T12:00:00Z",
        end="2026-10-23T12:01:00Z",
    )

    assert len(selected) == 1
    assert selected[0].line == 1


def test_naive_time_windows_report_the_missing_timezone() -> None:
    with pytest.raises(ValueError, match="timezone"):
        select_logs((), start="2026-10-23T08:32:00")


def test_save_case_preserves_each_iteration(tmp_path: Path) -> None:
    first = save_case(
        tmp_path,
        "sleep-case",
        report="First finding\n",
        case={"question": "Why did it sleep?"},
        result={"observed": "sleep"},
        script="print('first')\n",
    )
    second = save_case(
        tmp_path,
        "sleep-case",
        report="Revised finding\n",
        case={"question": "Why did it sleep?", "status": "tested"},
        result={"observed": "awake"},
        script="print('second')\n",
    )

    assert first.name == "0001"
    assert second.name == "0002"
    assert (first / "report.md").read_text() == "First finding\n"
    assert json.loads((first / "result.json").read_text()) == {"observed": "sleep"}
    assert (first / "investigate.py").read_text() == "print('first')\n"
    assert (second.parent.parent / "report.md").read_text() == "Revised finding\n"
    assert (
        json.loads((second.parent.parent / "case.json").read_text())["status"]
        == "tested"
    )
