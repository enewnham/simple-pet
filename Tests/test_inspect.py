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
            assert "Speak plainly." in rendered
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
