from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from .models import RelatedLink
from .wire import PullRequestData, RelatedLinkData


@dataclass(frozen=True)
class GitObservation:
    worktree: str
    branch: str
    head: str
    staged: int
    unstaged: int
    untracked: int
    conflicts: int
    upstream: str
    ahead: int
    behind: int
    repository: str = ""

    def data(self) -> GitObservationData:
        return {
            "worktree": self.worktree,
            "branch": self.branch,
            "head": self.head,
            "staged": self.staged,
            "unstaged": self.unstaged,
            "untracked": self.untracked,
            "conflicts": self.conflicts,
            "upstream": self.upstream,
            "ahead": self.ahead,
            "behind": self.behind,
            "repository": self.repository,
        }

    @property
    def pending(self) -> bool:
        return bool(
            self.staged
            or self.unstaged
            or self.untracked
            or self.conflicts
            or self.ahead
        )


@dataclass(frozen=True)
class PullRequest:
    number: int
    url: str
    state: str
    merged_at: str | None
    review: str
    checks: tuple[tuple[str, ...], ...]
    is_draft: bool | None = None
    head: str = ""
    head_repository: str = ""
    head_branch: str = ""
    review_requests: tuple[str, ...] = ()
    repository: str = ""
    check_revision: str = ""
    review_requested: bool | None = None
    title: str = ""
    body: str = ""
    base_branch: str = ""
    base_head: str = ""
    related_links: tuple[RelatedLink, ...] = ()
    related_links_error: str | None = None

    def data(self) -> PullRequestData:
        return {
            "number": self.number,
            "url": self.url,
            "state": self.state,
            "merged_at": self.merged_at,
            "review": self.review,
            "checks": self.checks,
            "is_draft": self.is_draft,
            "head": self.head,
            "head_repository": self.head_repository,
            "head_branch": self.head_branch,
            "review_requests": self.review_requests,
            "repository": self.repository,
            "check_revision": self.check_revision,
            "review_requested": self.review_requested,
            "title": self.title,
            "body": self.body,
            "base_branch": self.base_branch,
            "base_head": self.base_head,
            "related_links": tuple(
                related_link_data(link) for link in self.related_links
            ),
            "related_links_error": self.related_links_error,
        }

    @property
    def pending(self) -> bool:
        return self.state == "OPEN"


@dataclass(frozen=True)
class RepositoryObservation:
    subject: str
    worktree: str
    branch: str
    sessions: tuple[str, ...] = ()
    local: GitObservation | None = None
    pr: PullRequest | None = None
    current: bool = True
    revision: int = 0
    changed_at: float = 0
    local_checked_at: float = 0
    github_checked_at: float = 0
    github_attempted_at: float = 0
    local_error: str | None = None
    github_error: str | None = None
    repository: str = ""
    source: str = "repository"

    def packet(self) -> RepositoryObservationData:
        return {
            "subject": self.subject,
            "worktree": self.worktree,
            "branch": self.branch,
            "sessions": self.sessions,
            "local": self.local.data() if self.local else None,
            "pr": self.pr.data() if self.pr else None,
            "current": self.current,
            "revision": self.revision,
            "changed_at": self.changed_at,
            "local_checked_at": self.local_checked_at,
            "github_checked_at": self.github_checked_at,
            "github_attempted_at": self.github_attempted_at,
            "local_error": self.local_error,
            "github_error": self.github_error,
            "repository": self.repository,
            "source": self.source,
        }

    @property
    def observed_at(self) -> float:
        return max(self.local_checked_at, self.github_checked_at)

    @property
    def pending(self) -> bool:
        return bool(
            (self.current and self.local and self.local.pending)
            or (self.pr and self.pr.pending)
        )


@dataclass(frozen=True)
class RepositoryTarget:
    kind: str
    ref: str
    repository: str = ""
    checkout: str = ""
    include_links: bool = True


class GitObservationData(TypedDict):
    worktree: str
    branch: str
    head: str
    staged: int
    unstaged: int
    untracked: int
    conflicts: int
    upstream: str
    ahead: int
    behind: int
    repository: str


class RepositoryObservationData(TypedDict):
    subject: str
    worktree: str
    branch: str
    sessions: tuple[str, ...]
    local: GitObservationData | None
    pr: PullRequestData | None
    current: bool
    revision: int
    changed_at: float
    local_checked_at: float
    github_checked_at: float
    github_attempted_at: float
    local_error: str | None
    github_error: str | None
    repository: str
    source: str


def repository_observation_from_mapping(
    data: RepositoryObservationData,
) -> RepositoryObservation:
    local = data["local"]
    pr = data["pr"]
    return RepositoryObservation(
        subject=data["subject"],
        worktree=data["worktree"],
        branch=data["branch"],
        sessions=tuple(data["sessions"]),
        local=GitObservation(**local) if local else None,
        pr=pull_request_from_data(pr) if pr else None,
        current=data["current"],
        revision=data["revision"],
        changed_at=data["changed_at"],
        local_checked_at=data["local_checked_at"],
        github_checked_at=data["github_checked_at"],
        github_attempted_at=data["github_attempted_at"],
        local_error=data["local_error"],
        github_error=data["github_error"],
        repository=data["repository"],
        source=data["source"],
    )


def related_link_data(link: RelatedLink) -> RelatedLinkData:
    return {
        "kind": link.kind,
        "ref": link.ref,
        "relationship": link.relationship,
        "source": link.source,
        "stack": link.stack,
        "position": link.position,
        "state": link.state,
        "is_draft": link.is_draft,
        "head": link.head,
        "head_branch": link.head_branch,
        "title": link.title,
    }


def related_link_from_data(link: RelatedLinkData) -> RelatedLink:
    return RelatedLink(
        kind=link["kind"],
        ref=link["ref"],
        relationship=link["relationship"],
        source=link["source"],
        stack=link["stack"],
        position=link["position"],
        state=link["state"],
        is_draft=link["is_draft"],
        head=link["head"],
        head_branch=link["head_branch"],
        title=link["title"],
    )


def pull_request_from_data(pr: PullRequestData) -> PullRequest:
    return PullRequest(
        number=pr["number"],
        url=pr["url"],
        state=pr["state"],
        merged_at=pr["merged_at"],
        review=pr["review"],
        checks=tuple(tuple(check) for check in pr["checks"]),
        is_draft=pr["is_draft"],
        head=pr["head"],
        head_repository=pr["head_repository"],
        head_branch=pr["head_branch"],
        review_requests=tuple(pr["review_requests"]),
        repository=pr["repository"],
        check_revision=pr["check_revision"],
        review_requested=pr["review_requested"],
        title=pr["title"],
        body=pr["body"],
        base_branch=pr["base_branch"],
        base_head=pr["base_head"],
        related_links=tuple(
            related_link_from_data(link) for link in pr["related_links"]
        ),
        related_links_error=pr["related_links_error"],
    )
