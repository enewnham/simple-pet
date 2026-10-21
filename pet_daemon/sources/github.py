import json
from dataclasses import replace
from typing import Required, TypedDict
from urllib.parse import urlsplit

from ..work.models import RelatedLink
from ..work.observations import PullRequest
from .git import CommandPath, RunCommand, canonical_repository, command


class GitHubPullRequest(TypedDict, total=False):
    number: Required[int]
    url: Required[str]
    state: Required[str]
    mergedAt: str | None
    headRepositoryOwner: dict[str, str] | None
    headRepository: dict[str, str] | None
    headRefOid: str
    isDraft: bool
    headRefName: str
    title: str
    body: str
    baseRefName: str
    baseRefOid: str


def pull_request_identity(url: str) -> tuple[str, int | None]:
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.hostname
        and len(parts) == 4
        and parts[2] == "pull"
        and parts[3].isdigit()
    ):
        return canonical_repository(f"{parsed.hostname}/{parts[0]}/{parts[1]}"), int(
            parts[3]
        )
    return "", None


def _pull_request(data: GitHubPullRequest) -> PullRequest:
    repository, _ = pull_request_identity(data["url"])
    owner = data.get("headRepositoryOwner") or {}
    head_repository = data.get("headRepository") or {}
    name = head_repository.get("nameWithOwner")
    if not name and owner.get("login") and head_repository.get("name"):
        name = f"{owner['login']}/{head_repository['name']}"
    host = urlsplit(data["url"]).hostname
    head_identity = canonical_repository(f"{host}/{name}") if name and host else ""
    return PullRequest(
        number=data["number"],
        url=data["url"],
        state=data["state"],
        merged_at=data.get("mergedAt"),
        is_draft=data.get("isDraft"),
        head=data.get("headRefOid") or "",
        head_repository=head_identity,
        head_branch=data.get("headRefName") or "",
        repository=repository,
        title=data.get("title") or "",
        body=data.get("body") or "",
        base_branch=data.get("baseRefName") or "",
        base_head=data.get("baseRefOid") or "",
    )


async def _stack_links(
    pr: PullRequest, path: CommandPath, run: RunCommand
) -> PullRequest:
    host, slug = pr.repository.split("/", 1)
    try:
        output = await run(
            "gh",
            "api",
            f"repos/{slug}/stacks?pull_request={pr.number}",
            "--hostname",
            host,
            cwd=path,
        )
        stacks = json.loads(output)
        matches = [
            stack
            for stack in stacks
            if any(item["number"] == pr.number for item in stack["pull_requests"])
        ]
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
            links.append(
                RelatedLink(
                    kind="pull_request",
                    ref=f"https://{pr.repository}/pull/{item['number']}",
                    relationship="stack_entry",
                    source="github_stack",
                    stack=stack["number"],
                    position=position,
                    state=state,
                    is_draft=item.get("draft"),
                    head=(item.get("head") or {}).get("sha", ""),
                    head_branch=(item.get("head") or {}).get("ref", ""),
                )
            )
        return replace(pr, related_links=tuple(links))
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        TimeoutError,
    ) as error:
        return replace(pr, related_links_error=str(error)[:240])


async def observe(
    path: CommandPath,
    branch: str,
    number: int | None = None,
    run: RunCommand = command,
    repository: str = "",
    url: str = "",
    include_links: bool = True,
) -> PullRequest | None:
    fields = (
        "number,url,state,mergedAt,isDraft,headRefOid,"
        "headRepository,headRepositoryOwner,headRefName,title,body,baseRefName,baseRefOid"
    )
    if number is not None or url:
        args = ["gh", "pr", "view", url or str(number), "--json", fields]
        if not url and repository and not repository.startswith("local:"):
            args.extend(("--repo", repository))
        data = json.loads(await run(*args, cwd=path))
        pr = _pull_request(data)
        return await _stack_links(pr, path, run) if include_links else pr
    if not repository or repository.startswith("local:"):
        raise RuntimeError("GitHub repository identity is unavailable")
    metadata = json.loads(
        await run("gh", "repo", "view", repository, "--json", "isFork,parent", cwd=path)
    )
    query_repository = repository
    if metadata.get("isFork"):
        parent = metadata.get("parent") or {}
        owner = (parent.get("owner") or {}).get("login")
        name = parent.get("nameWithOwner") or (
            f"{owner}/{parent['name']}" if owner and parent.get("name") else ""
        )
        if name:
            host = repository.split("/", 1)[0]
            query_repository = canonical_repository(f"{host}/{name}")
    output = await run(
        "gh",
        "pr",
        "list",
        "--head",
        branch,
        "--state",
        "all",
        "--limit",
        "100",
        "--repo",
        query_repository,
        "--json",
        fields,
        cwd=path,
    )
    records = json.loads(output)
    candidates = [_pull_request(item) for item in records]
    if any(not item.head_repository or not item.head_branch for item in candidates):
        raise RuntimeError("Pull request head identity is unavailable")
    matches = [
        item
        for item in candidates
        if item.head_repository == repository and item.head_branch == branch
    ]
    opened = [item for item in matches if item.pending]
    if len(opened) > 1:
        raise RuntimeError(
            "Multiple open pull requests match the repository and branch"
        )
    if not opened and len(matches) > 1:
        raise RuntimeError(
            "Multiple historical pull requests match the repository and branch"
        )
    pr = opened[0] if opened else matches[0] if matches else None
    return await _stack_links(pr, path, run) if pr and include_links else pr
