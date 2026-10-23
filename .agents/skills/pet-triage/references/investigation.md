# Small investigations

Run focused Python from the repository root with `UV_CACHE_DIR=.cache/uv uv run --locked python -`. Reuse saved artifacts for follow-ups. The report template supplies the layout; the investigation script supplies findings and links to evidence.

## Capture once, inspect an index

For a delivery case, supply `message_id`, `log_file`, and `case_dir` to this recipe. It resumes the captured chain when available. Full records retain their original path and line even after log rotation.

```python
import json
from dataclasses import asdict

from pet_daemon.logging import json_default
from pet_daemon.triage import load_logs, trace_delivery

capture = case_dir / "captured.json"
if capture.exists():
    bundle = json.loads(capture.read_text())
else:
    logs = load_logs(log_file)
    trace = trace_delivery(logs.entries, message_id)
    bundle = {
        "delivery": asdict(trace),
        "log_diagnostics": [asdict(item) for item in logs.diagnostics],
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    capture.write_text(json.dumps(bundle, default=json_default, indent=2))

records = bundle["delivery"]["chain"]
print(
    json.dumps(
        {
            "diagnostics": bundle["delivery"]["diagnostics"],
            "malformed_lines": len(bundle["log_diagnostics"]),
            "records": [
                {
                    "index": index,
                    "time": item["data"].get("timestamp"),
                    "event": item["data"].get("event"),
                    "purpose": item["data"].get("reason"),
                    "request_bytes": item["data"].get("request_bytes"),
                    "source": f"{item['source']}:{item['line']}",
                }
                for index, item in enumerate(records)
            ],
        }
    )
)
```

Inspect a selected record's `data`, then only the fields needed for the question. A model request's `messages[0].content` is JSON. Save full request and response records beside the report and link them from the inspector.

For other behavior, use `select_logs` with exact fields and a timezone-aware window. For upstream hooks, filter `event="hook.received"` and `packet={"session": native_session_id}`. For session snapshots, select the matching session before printing. Expand the saved evidence when the causal path leaves the captured chain.

## Choose the smallest experiment

| Question | Starting point | Evidence produced |
| --- | --- | --- |
| What did the model see? | Captured `observation.request` | Exact system instructions and payload |
| Where did a supplied fact originate? | Related hooks and stored observations | Input → state → prompt trace |
| What would current context construction produce? | Production prompt builder with captured or reconstructed inputs | Request diff; no model call needed |
| Would scheduling, reactions, or retries change? | `ReplayHarness.advance` and renderer feedback | Timed decisions and emitted packets |
| Would wording improve? | Existing model adapter with a copied captured request | One live sample and usage |

Check model availability once per investigation before building live variants. Preserve an unavailable result as a capability limit. Use captured outputs or deterministic experiments for questions they can answer.

Compare captured daemon startup/configuration and store schema with the working tree before interpreting a current-code replay as historical reproduction. Record meaningful differences in the case.

## Render saved results

Keep evidence collection, experiments, and rendering separately callable. Load the chosen iteration's `case.json` and `result.json`, fill the [inspector template](../assets/report.md), and save a new iteration with `save_case`. A wording or layout revision reuses those results. Keep new output beside its baseline when an experiment runs again.
