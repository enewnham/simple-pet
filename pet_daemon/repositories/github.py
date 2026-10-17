from dataclasses import dataclass
import json

from .git import command


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str
    state: str
    merged_at: str | None
    review: str
    checks: tuple

    @property
    def pending(self):
        return self.state == "OPEN"


async def observe(path, branch, number=None, run=command):
    fields = "number,url,state,mergedAt,reviewDecision,statusCheckRollup"
    if number is None:
        output = await run("gh", "pr", "list", "--head", branch, "--state", "all", "--limit", "10", "--json", fields, cwd=path)
        candidates = json.loads(output)
        if not candidates:
            return None
        data = max(candidates, key=lambda item: (item["state"] == "OPEN", item["number"]))
    else:
        data = json.loads(await run("gh", "pr", "view", str(number), "--json", fields, cwd=path))
    checks = tuple(sorted((item.get("name") or item.get("context", ""),
                           item.get("conclusion") or item.get("state") or item.get("status", ""))
                          for item in data.get("statusCheckRollup") or []))
    return PullRequest(data["number"], data["url"], data["state"], data.get("mergedAt"),
                       data.get("reviewDecision") or "", checks)
