import asyncio
import json
from io import StringIO
from pathlib import Path

from rich.console import Console
from textual.widgets import OptionList, Static

from pet_daemon.inspect import RequestInspector, commentary_requests
from pet_daemon.triage import load_logs


def write_logs(path: Path) -> None:
    records = [
        {"event": "observation.request", "reason": "story", "request": {}},
        {
            "event": "observation.request",
            "request": {
                "model": "claude-test",
                "system": [{"type": "text", "text": "Speak **plainly**."}],
                "messages": [{"role": "user", "content": '{"goal":"Fix parser"}'}],
            },
        },
        {
            "event": "observation.request",
            "observation_id": "quiet",
            "request": {"system": [], "messages": []},
        },
        {
            "event": "observation.response",
            "reason": "story",
            "response": {"content": [{"type": "text", "text": '{"story":"work"}'}]},
        },
        {
            "event": "observation.response",
            "response": {
                "content": [{"type": "text", "text": '{"speak":"Parser fixed."}'}]
            },
        },
        {
            "event": "observation.response",
            "observation_id": "quiet",
            "response": {"content": [{"type": "text", "text": '{"speak":null}'}]},
        },
        {
            "event": "renderer.command",
            "packet": {"delivery_id": "bubble", "text": "Parser fixed."},
        },
        {"event": "observation.finished", "outcome": "submitted"},
        {
            "event": "speech.delivery",
            "observation_id": None,
            "message": "bubble",
            "outcome": "accepted",
        },
        {
            "event": "speech.delivery",
            "run_id": "another-run",
            "message": "bubble",
            "outcome": "failed",
        },
    ]
    path.write_text(
        "".join(
            json.dumps(
                {
                    "run_id": "run",
                    "observation_id": "spoken",
                    "reason": "check_in",
                    "timestamp": "2026-10-23T12:00:00Z",
                    **record,
                }
            )
            + "\n"
            for record in records
        )
    )


def test_timeline_pairs_interleaved_commentary_with_displayed_bubbles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pet.jsonl"
    write_logs(path)
    lines = path.read_text().splitlines(keepends=True)
    path.with_suffix(".jsonl.1").write_text("".join(lines[:4]))
    path.write_text("".join(lines[4:]))
    items = commentary_requests(load_logs(path).entries)

    assert [item.key for item in items] == [("run", "quiet"), ("run", "spoken")]
    assert items[0].speech is None
    assert items[0].outcome == "silent"
    assert items[1].speech == "Parser fixed."
    assert items[1].outcome == "displayed"
    assert items[1].request["messages"][0]["content"] == '{"goal":"Fix parser"}'


def test_viewer_selects_requests_and_refreshes_without_losing_selection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pet.jsonl"
    write_logs(path)

    async def browse() -> None:
        app = RequestInspector(path)
        async with app.run_test(size=(130, 40)) as pilot:
            await pilot.pause()
            timeline = app.query_one(OptionList)
            assert timeline.highlighted == 0
            await pilot.press("down")
            assert timeline.highlighted == 1
            output = StringIO()
            console = Console(file=output, width=100)
            for pane in ("bubble", "system", "messages"):
                console.print(app.query_one(f"#{pane}", Static).content)
            rendered = output.getvalue()
            assert "Parser fixed." in rendered
            assert "Speak **plainly**." in rendered
            assert '  "goal": "Fix parser"' in rendered
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "observation.request",
                            "run_id": "run",
                            "observation_id": "new",
                            "timestamp": "2026-10-23T12:01:00Z",
                            "request": {"system": [], "messages": []},
                        }
                    )
                    + "\n"
                )
            await pilot.press("r")
            await pilot.pause()
            assert timeline.highlighted == 2
            assert app.requests[timeline.highlighted].outcome == "displayed"
            await pilot.press("tab", "tab", "pagedown", "q")

    asyncio.run(browse())


def test_correction_retains_each_attempt_and_aggregates_usage(tmp_path: Path) -> None:
    path = tmp_path / "pet.jsonl"
    write_logs(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    request = next(
        record for record in records if record.get("request", {}).get("model")
    )
    request["request"]["messages"][0]["content"] = (
        "<task>Fix parser</task>\n<evidence>Use &lt;tag&gt; safely</evidence>"
    )
    response_index = next(
        index
        for index, record in enumerate(records)
        if record["event"] == "observation.response" and record["reason"] != "story"
    )
    base = {
        key: request[key] for key in ("run_id", "observation_id", "reason", "timestamp")
    }
    records[response_index:response_index] = [
        {
            **base,
            "event": "observation.response",
            "response": {
                "content": [{"type": "text", "text": '{"speak":"Repeated speech."}'}]
            },
        },
        {
            **base,
            "event": "observation.usage",
            "input_tokens": 10,
            "output_tokens": 2,
            "duration_seconds": 0.25,
        },
        {
            **base,
            "event": "observation.request",
            "request": {
                "model": "claude-test",
                "system": [],
                "messages": [{"role": "user", "content": "Avoid repeated speech."}],
            },
        },
    ]
    records.insert(
        response_index + 4,
        {
            **base,
            "event": "observation.usage",
            "input_tokens": 20,
            "output_tokens": 3,
            "duration_seconds": 0.5,
        },
    )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    items = commentary_requests(load_logs(path).entries)
    spoken = next(item for item in items if item.key[1] == "spoken")
    assert len(items) == 2
    assert len(spoken.attempts) == 2
    assert spoken.usage == {
        "input_tokens": 30,
        "output_tokens": 5,
        "duration_seconds": 0.75,
    }
    assert spoken.outcome == "displayed"
    assert spoken.speech == "Parser fixed."

    async def browse() -> None:
        app = RequestInspector(path)
        async with app.run_test(size=(130, 40)) as pilot:
            await pilot.pause()
            app.show_request(spoken)
            output = StringIO()
            console = Console(file=output, width=120)
            for pane in ("bubble", "system", "messages"):
                console.print(app.query_one(f"#{pane}", Static).content)
            rendered = output.getvalue()
            assert "<task>Fix parser</task>" in rendered
            assert "<evidence>Use &lt;tag&gt; safely</evidence>" in rendered
            assert "Repeated speech." in rendered
            assert "Parser fixed." in rendered
            assert "Avoid repeated speech." in rendered
            assert "2 attempts" in rendered
            assert "30 in / 5 out · 0.75s" in rendered
            assert '"input_tokens": 10' in rendered
            assert '"input_tokens": 20' in rendered

    asyncio.run(browse())
