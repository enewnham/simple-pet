import json
import logging
import os
from pathlib import Path
import tempfile

from .telemetry import trace


log = logging.getLogger("pet-daemon")


def load_memory(path, pet):
    try:
        data = json.loads(path.read_text())
        if data["pet"] != pet:
            trace("memory.skipped", reason="different_pet")
            return None, [], []
        history = data["messages"]
        summary = data.get("summary")
        summaries = data.get("summaries", [])
        trace("memory.loaded", history_pairs=len(history) // 2, summaries=len(summaries))
        return summary, summaries, history
    except FileNotFoundError:
        return None, [], []
    except (OSError, ValueError, KeyError, TypeError) as error:
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
    except (OSError, ValueError) as error:
        log.warning("Could not save pet memory: %s", error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                log.warning("Could not remove temporary pet memory: %s", error)
