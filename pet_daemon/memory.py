import json
import logging
import math
import os
from pathlib import Path
import tempfile

from .telemetry import trace


log = logging.getLogger("pet-daemon")


def validate_memory(summary, summaries, history):
    if not isinstance(history, list) or len(history) % 2:
        raise ValueError("Memory requires complete conversation exchanges")
    for index, message in enumerate(history):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(message, dict) or message.get("role") != role or not isinstance(message.get("content"), str):
            raise ValueError("Memory requires alternating user and assistant messages")
        if role == "user":
            record = json.loads(message["content"])
            timestamp = record["observed_at"]
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                raise ValueError("Memory requires finite observation timestamps")
    if not isinstance(summaries, list):
        raise ValueError("Memory summaries must be a list")
    groups = (("long_term_memory", [summary] if summary is not None else []), ("compacted_memory", summaries))
    for kind, messages in groups:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user" or not isinstance(message.get("content"), str):
                raise ValueError("Memory summaries require user messages")
            record = json.loads(message["content"])
            text, timestamp = record[kind], record["through_observed_at"]
            if not isinstance(text, str) or not text.strip() or type(timestamp) not in (int, float) or not math.isfinite(timestamp):
                raise ValueError("Memory summaries require text and finite timestamps")


def load_memory(path, pet):
    try:
        data = json.loads(path.read_text())
        if data["pet"] != pet:
            trace("memory.skipped", reason="different_pet")
            return None, [], []
        history = data["messages"]
        summary = data.get("summary")
        summaries = data.get("summaries", [])
        validate_memory(summary, summaries, history)
        trace("memory.loaded", history_pairs=len(history) // 2, summaries=len(summaries))
        return summary, summaries, history
    except FileNotFoundError:
        return None, [], []
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        log.warning("Could not load pet memory: %s", error)
        return None, [], []


def save_memory(path, pet, summary, summaries, history):
    temporary = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pet": pet, "summary": summary, "summaries": summaries, "messages": history}, stream, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)
        trace("memory.saved", history_pairs=len(history) // 2)
        return True
    except (OSError, ValueError) as error:
        log.warning("Could not save pet memory: %s", error)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                log.warning("Could not remove temporary pet memory: %s", error)
