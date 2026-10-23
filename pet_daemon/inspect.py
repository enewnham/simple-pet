"""Browse commentary requests from the daemon's JSONL logs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from rich.console import Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, OptionList, Static

from .triage import LogEntry, load_logs


@dataclass
class CommentaryRequest:
    entry: LogEntry
    request: dict[str, Any]
    speech: str | None = None
    outcome: str = "pending"
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.entry.data["run_id"], self.entry.data["observation_id"]

    @property
    def label(self) -> str:
        stamp = datetime.fromisoformat(self.entry.data["timestamp"]).astimezone()
        reason = self.entry.data.get("reason", "commentary")
        preview = " ".join((self.speech or "No speech").split())
        return f"{stamp:%m/%d %H:%M:%S} · {reason}\n{self.outcome} · {preview[:100]}"


def commentary_requests(entries: Sequence[LogEntry]) -> list[CommentaryRequest]:
    """Join commentary payloads and renderer outcomes across interleaved observations."""
    requests: dict[tuple[str, str], CommentaryRequest] = {}
    deliveries: dict[tuple[str, str], CommentaryRequest] = {}
    for entry in entries:
        data = entry.data
        key = (data.get("run_id"), data.get("observation_id"))
        event = data.get("event")
        if event == "observation.request" and data.get("reason") != "story":
            item = CommentaryRequest(entry, data["request"])
            requests[item.key] = item
        item = requests.get(key)
        if event == "speech.delivery":
            delivery = deliveries.get((data.get("run_id"), data.get("message")))
            if delivery is not None:
                delivery.outcome = (
                    "displayed" if data["outcome"] == "accepted" else data["outcome"]
                )
            continue
        if item is None:
            continue
        if event == "observation.response" and data.get("reason") != "story":
            content = data["response"].get("content", [])
            text = "".join(
                block["text"] for block in content if block["type"] == "text"
            )
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                item.outcome = "invalid response"
            else:
                if isinstance(result, dict) and isinstance(result.get("speak"), str):
                    item.speech = result["speak"]
                    item.outcome = "generated"
                else:
                    item.outcome = "silent"
        elif event == "observation.usage" and data.get("reason") != "story":
            item.usage = data
        elif event == "observation.finished":
            item.outcome = data.get("outcome", item.outcome)
        elif event == "renderer.command":
            packet = data.get("packet", {})
            if "delivery_id" in packet:
                deliveries[(data["run_id"], packet["delivery_id"])] = item
    return list(reversed(requests.values()))


def render_content(content: object) -> Group:
    """Render text blocks as Markdown and JSON payloads with indentation."""
    if isinstance(content, list):
        return Group(*(render_content(block) for block in content))
    if isinstance(content, dict) and content.get("type") == "text":
        return render_content(content["text"])
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return Group(Markdown(content))
    else:
        value = content
    return Group(
        Syntax(
            json.dumps(value, indent=2, ensure_ascii=False),
            "json",
            theme="ansi_light",
            word_wrap=True,
        )
    )


class RequestInspector(App[None]):
    TITLE = "Pet commentary"
    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Refresh logs"),
    ]
    CSS = """
    #timeline { width: 36%; min-width: 26; }
    #detail { width: 1fr; }
    #bubble { height: auto; max-height: 12; border: round $accent; padding: 0 1; }
    #system-pane { height: 1fr; min-height: 6; }
    #messages-pane { height: 2fr; min-height: 6; }
    .pane { border: round $primary; padding: 0 1; }
    """

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.theme = "textual-ansi"
        self.path = path
        self.requests: list[CommentaryRequest] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield OptionList(id="timeline")
            with Vertical(id="detail"):
                yield Static("No commentary requests. Press r to refresh.", id="bubble")
                with VerticalScroll(id="system-pane", classes="pane"):
                    yield Static(id="system")
                with VerticalScroll(id="messages-pane", classes="pane"):
                    yield Static(id="messages")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#timeline").border_title = "Timeline · newest first"
        self.query_one("#bubble").border_title = "Speech bubble"
        self.query_one("#system-pane").border_title = "System prompt"
        self.query_one("#messages-pane").border_title = "Messages sent"
        self.action_reload()
        self.query_one(OptionList).focus()

    def action_reload(self) -> None:
        timeline = self.query_one(OptionList)
        highlighted = timeline.highlighted
        selected = (
            self.requests[highlighted].key
            if highlighted is not None and self.requests
            else None
        )
        try:
            evidence = load_logs(self.path)
        except OSError as error:
            self.notify(str(error), severity="error")
            return
        self.requests = commentary_requests(evidence.entries)
        timeline.clear_options()
        timeline.add_options(Text(item.label) for item in self.requests)
        self.sub_title = f"{len(self.requests)} requests · {self.path.name}"
        if evidence.diagnostics:
            self.notify(f"Skipped {len(evidence.diagnostics)} malformed log lines")
        if self.requests:
            index = next(
                (i for i, item in enumerate(self.requests) if item.key == selected), 0
            )
            timeline.highlighted = index
            self.show_request(self.requests[index])
        else:
            self.query_one("#bubble", Static).update(
                "No commentary requests. Press r to refresh."
            )
            self.query_one("#system", Static).update("")
            self.query_one("#messages", Static).update("")

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        self.show_request(self.requests[event.option_index])

    def show_request(self, item: CommentaryRequest) -> None:
        data = item.entry.data
        usage = item.usage
        metadata = (
            f"{item.outcome} · {data.get('reason')} · {item.request.get('model')}\n"
            f"{usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out"
            f" · {usage.get('duration_seconds', '?')}s"
        )
        self.query_one("#bubble", Static).update(
            Group(
                Text(item.speech or "No speech", style="bold"),
                Text(metadata, style="dim"),
            )
        )
        self.query_one("#system", Static).update(
            render_content(item.request.get("system"))
        )
        self.query_one("#messages", Static).update(
            Group(
                *(
                    Group(
                        Text(message["role"].upper(), style="bold cyan"),
                        render_content(message["content"]),
                    )
                    for message in item.request.get("messages", [])
                )
            )
        )
        self.query_one("#system-pane", VerticalScroll).scroll_home(animate=False)
        self.query_one("#messages-pane", VerticalScroll).scroll_home(animate=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "logs/pet-daemon.jsonl",
        help="JSONL log path (includes numbered rotations)",
    )
    args = parser.parse_args()
    RequestInspector(args.log.expanduser()).run()


if __name__ == "__main__":
    main()
