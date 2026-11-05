import json
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    request = json.load(sys.stdin)
    with (root / "calls.jsonl").open("a") as stream:
        stream.write(json.dumps(request) + "\n")
    mode_path = root / "mode"
    mode = mode_path.read_text().strip() if mode_path.exists() else "ready"
    task_id, command_id = request["task_id"], request["command_id"]
    receipt = root / (command_id + ".json")
    operation = request["operation"]
    token = "native-input-2" if mode == "token_changed" else "native-input-1"
    if operation == "inspect" and receipt.exists():
        result = json.loads(receipt.read_text())
        result["status"] = "ready"
        result["readiness"] = "ready" if mode != "unknown" else "unknown"
    else:
        result = {
            "protocol_version": 1,
            "task_id": task_id,
            "command_id": command_id,
            "outcome": "accepted",
            "observed_at": 1000,
            "status": "preparing"
            if operation == "launch"
            else "ready"
            if operation == "inspect"
            else "stopped"
            if operation == "stop"
            else "running",
            "readiness": "ready" if operation == "inspect" else "unknown",
            "interaction_revision": token,
            "external_id": "external-" + task_id,
            "session": {
                "adapter": "fixture",
                "harness": "codex",
                "session": "native-" + task_id,
                "workspace": request["brief"]["repository"] + "/worktree",
                "external_id": "external-" + task_id,
            },
        }
    if mode == "preparation_failed":
        result.update(
            outcome="failed",
            status="unknown",
            session=None,
            error="Workspace preparation failed",
        )
    if mode == "unknown" and operation == "inspect":
        result["readiness"] = "unknown"
    if mode == "no_token":
        result["interaction_revision"] = None
    if operation == "continue" and request["expected_interaction_revision"] != token:
        result.update(
            outcome="failed",
            error="Native input changed before delivery",
            readiness="unknown",
        )
    if operation not in ("inspect", "open"):
        if not receipt.exists() and result["outcome"] == "accepted":
            with (root / "effects.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"command_id": command_id, "operation": operation})
                    + "\n"
                )
        receipt.write_text(json.dumps(result))
    if mode == "timeout" and operation != "inspect":
        time.sleep(30)
    if mode == "wait" and operation != "inspect":
        (root / "entered").write_text(command_id)
        while not (root / "release").exists():
            time.sleep(0.005)
    if mode == "malformed" and operation != "inspect":
        print("invalid result")
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
