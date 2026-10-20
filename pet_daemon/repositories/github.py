from dataclasses import dataclass, replace
import json
from urllib.parse import urlsplit

from .git import canonical_repository, command


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str
    state: str
    merged_at: str | None
    review: str
    checks: tuple
    is_draft: bool | None = None
    head: str = ""
    head_repository: str = ""
    head_branch: str = ""
    review_requests: tuple = ()
    repository: str = ""
    check_revision: str = ""
    review_requested: bool | None = None
    title: str = ""
    body: str = ""
    base_branch: str = ""
    base_head: str = ""
    related_links: tuple = ()
    related_links_error: str | None = None

    @property
    def pending(self):
        return self.state == "OPEN"


def pull_request_identity(url):
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname and len(parts) == 4 and parts[2] == "pull" and parts[3].isdigit():
        return canonical_repository(f"{parsed.hostname}/{parts[0]}/{parts[1]}"), int(parts[3])
    return "", None


def _pull_request(data, viewer=""):
    repository, _ = pull_request_identity(data["url"])
    owner = data.get("headRepositoryOwner") or {}
    head_repository = data.get("headRepository") or {}
    name = head_repository.get("nameWithOwner")
    if not name and owner.get("login") and head_repository.get("name"):
        name = f'{owner["login"]}/{head_repository["name"]}'
    host = urlsplit(data["url"]).hostname
    head_identity = canonical_repository(f"{host}/{name}") if name and host else ""
    checks = []
    for item in data.get("statusCheckRollup") or []:
        check = (item.get("name") or item.get("context", ""),
                 item.get("conclusion") or item.get("state") or item.get("status", ""))
        source = item.get("source") or (item.get("app") or {}).get("slug") or item.get("__typename") or ""
        sha = item.get("headSha") or item.get("sha") or (item.get("checkSuite") or {}).get("headSha") or ""
        if source or sha:
            check += (source, sha)
        checks.append(check)
    requests = tuple(sorted(item.get("login") or item.get("slug") or item.get("name") or ""
                            for item in data.get("reviewRequests") or []))
    reviewers = data.get("reviewRequests") or []
    review_requested = False if "reviewRequests" in data and not reviewers else None
    if viewer:
        requested = any((item.get("login") or "").casefold() == viewer.casefold() for item in reviewers)
        review_requested = requested if requested or all(item.get("login") for item in reviewers) else None
    head = data.get("headRefOid") or ""
    return PullRequest(data["number"], data["url"], data["state"], data.get("mergedAt"),
                       data.get("reviewDecision") or "", tuple(sorted(checks)), data.get("isDraft"),
                       head, head_identity, data.get("headRefName") or "", requests, repository, head, review_requested,
                       data.get("title") or "", data.get("body") or "", data.get("baseRefName") or "", data.get("baseRefOid") or "")


async def _viewer(records, path, run):
    if not any(item.get("login") for data in records for item in data.get("reviewRequests") or []):
        return ""
    try:
        return (await run("gh", "api", "user", "--jq", ".login", cwd=path)).strip()
    except (OSError, RuntimeError, TimeoutError):
        return ""


async def _stack_links(pr, path, run):
    host, slug = pr.repository.split("/", 1)
    try:
        output = await run("gh", "api", f"repos/{slug}/stacks?pull_request={pr.number}", "--hostname", host, cwd=path)
        stacks = json.loads(output)
        matches = [stack for stack in stacks if any(item["number"] == pr.number for item in stack["pull_requests"])]
        if len(matches) > 1:
            raise RuntimeError("Pull request belongs to multiple stacks")
        if not matches:
            return pr
        stack = matches[0]
        links = []
        for position, item in enumerate(stack["pull_requests"], 1):
            if item["number"] == pr.number:
                continue
            state = "MERGED" if item.get("merged_at") else item["state"].upper()
            links.append({"kind": "pull_request", "ref": f'https://{pr.repository}/pull/{item["number"]}',
                          "relationship": "stack_entry", "source": "github_stack", "stack": stack["number"],
                          "position": position, "state": state, "is_draft": item.get("draft"),
                          "head": (item.get("head") or {}).get("sha", ""),
                          "head_branch": (item.get("head") or {}).get("ref", "")})
        return replace(pr, related_links=tuple(links))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, TimeoutError) as error:
        return replace(pr, related_links_error=str(error)[:240])


async def observe(path, branch, number=None, run=command, repository="", url="", include_links=True):
    fields = ("number,url,state,mergedAt,reviewDecision,statusCheckRollup,isDraft,headRefOid,"
              "headRepository,headRepositoryOwner,headRefName,reviewRequests,title,body,baseRefName,baseRefOid")
    if number is not None or url:
        args = ["gh", "pr", "view", url or str(number), "--json", fields]
        if not url and repository and not repository.startswith("local:"):
            args.extend(("--repo", repository))
        data = json.loads(await run(*args, cwd=path))
        pr = _pull_request(data, await _viewer([data], path, run))
        return await _stack_links(pr, path, run) if include_links else pr
    if not repository or repository.startswith("local:"):
        raise RuntimeError("GitHub repository identity is unavailable")
    metadata = json.loads(await run("gh", "repo", "view", repository, "--json", "isFork,parent", cwd=path))
    query_repository = repository
    if metadata.get("isFork"):
        parent = metadata.get("parent") or {}
        owner = (parent.get("owner") or {}).get("login")
        name = parent.get("nameWithOwner") or (f'{owner}/{parent["name"]}' if owner and parent.get("name") else "")
        if name:
            host = repository.split("/", 1)[0]
            query_repository = canonical_repository(f"{host}/{name}")
    output = await run("gh", "pr", "list", "--head", branch, "--state", "all", "--limit", "100",
                       "--repo", query_repository, "--json", fields, cwd=path)
    records = json.loads(output)
    viewer = await _viewer(records, path, run)
    candidates = [_pull_request(item, viewer) for item in records]
    if any(not item.head_repository or not item.head_branch for item in candidates):
        raise RuntimeError("Pull request head identity is unavailable")
    matches = [item for item in candidates if item.head_repository == repository and item.head_branch == branch]
    opened = [item for item in matches if item.pending]
    if len(opened) > 1:
        raise RuntimeError("Multiple open pull requests match the repository and branch")
    pr = opened[0] if opened else max(matches, key=lambda item: item.number) if matches else None
    return await _stack_links(pr, path, run) if pr and include_links else pr
