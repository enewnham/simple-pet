"""Send agent events and commentary requests to the pet daemon."""

import argparse
import json
import math
import os
import socket
import sys
import time
from collections.abc import Mapping

from ..domain import SessionSource
from ..sessions.processes import discover_owner
from ..transport import MAX_DATAGRAM_BYTES, SOCKET_BUFFER_BYTES


def encode(payload: Mapping[str, object]) -> bytes | None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return data if len(data) <= MAX_DATAGRAM_BYTES else None


def send(payload: Mapping[str, object]) -> bool:
    if payload.get("kind") == "event" and "capture_at" not in payload:
        payload = dict(payload, capture_at=time.time())
    data = encode(payload)
    if data is None:
        print(
            f"pet-notify: message exceeds {MAX_DATAGRAM_BYTES} bytes", file=sys.stderr
        )
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_BYTES)
            client.setblocking(False)
            client.sendto(
                data, os.environ.get("PET_DAEMON_SOCKET", "/tmp/pet-daemon.sock")
            )
        return True
    except OSError:
        return False


def bubble_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(prog="pet-notify", description=__doc__)
    parser.add_argument("--session")
    parser.add_argument("--status", choices=("start", "done", "error"))
    parser.add_argument("--call", default="single")
    parser.add_argument("--tool")
    parser.add_argument("--exit-code", type=int)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser(
        "clear-memory",
        help="clear the running pet's saved knowledge and tracked sessions",
    )
    commands.add_parser(
        "clear-speech",
        help="forget recent remarks and dismiss the current speech bubble",
    )
    intent = commands.add_parser("intent", help="record a session's goal")
    intent.add_argument("--session", required=True)
    intent.add_argument("--text", required=True)
    speech = commands.add_parser("speak", help="request asynchronous commentary")
    speech.add_argument("--context", required=True)
    speech.add_argument("--ttl", type=bubble_seconds)
    args = parser.parse_args()
    payload: dict[str, object]
    if args.command == "clear-memory":
        payload = {"kind": "clear_memory"}
    elif args.command == "clear-speech":
        payload = {"kind": "clear_speech"}
    elif args.command == "speak":
        payload = {"kind": "speak", "context": args.context}
        if args.ttl is not None:
            payload["ttl"] = args.ttl
    elif args.command == "intent":
        payload = {
            "kind": "event",
            "source": "cli",
            "session": args.session,
            "event": "intent",
            "goal": args.text,
            "manual": True,
        }
    else:
        if not args.session or not args.status:
            parser.error("--session and --status are required for tool events")
        payload = {
            "kind": "event",
            "source": "cli",
            "session": args.session,
            "event": {
                "start": "tool_start",
                "done": "tool_done",
                "error": "tool_error",
            }[args.status],
            "call": args.call,
            "manual": True,
        }
        if args.tool:
            payload["tool"] = args.tool
        if args.exit_code is not None:
            payload["exit_code"] = args.exit_code
        if args.status != "start":
            payload["outcome"] = "error" if args.status == "error" else "success"
    if payload.get("kind") == "event":
        owner = discover_owner(SessionSource.COMMAND)
        if owner:
            payload["owner"] = owner.packet()
    sent = send(payload)
    if args.command in ("clear-memory", "clear-speech") and not sent:
        parser.exit(1, f"pet-notify: could not send {args.command} to the pet daemon\n")


if __name__ == "__main__":
    main()
