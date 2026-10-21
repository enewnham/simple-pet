from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from .models import ExecutionEvidence, IntentEvidence, ProgressEvidence, RelatedLink
from .wire import PullRequestData, RelatedLinkData


@dataclass(frozen=True)
class SessionObservation:
    session: str
    aliases: tuple[str, ...] = ()
    parent: str | None = None
    location: str | None = None
    reports: tuple[IntentEvidence | ProgressEvidence | ExecutionEvidence, ...] = ()
    repository: str | None = None
    branch: str | None = None
    pull_request: str | None = None


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
    is_draft: bool | None = None
    head: str = ""
    head_repository: str = ""
    head_branch: str = ""
    repository: str = ""
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
            "is_draft": self.is_draft,
            "head": self.head,
            "head_repository": self.head_repository,
            "head_branch": self.head_branch,
            "repository": self.repository,
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
class CheckoutObservation:
    worktree: str
    local: GitObservation | None
    checked_at: float
    error: str | None = None
    available: bool = True
    last_success_at: float | None = None

    def packet(self) -> CheckoutObservationData:
        return {
            "kind": "checkout",
            "worktree": self.worktree,
            "local": self.local.data() if self.local else None,
            "checked_at": self.checked_at,
            "error": self.error,
            "last_success_at": self.last_success_at,
            "available": self.available,
        }


@dataclass(frozen=True)
class BranchObservation:
    repository: str
    branch: str
    pr: PullRequest | None
    checked_at: float
    error: str | None = None
    last_success_at: float | None = None
    # Retain observed PR resolutions so branch reuse cannot reattribute reports.
    resolutions: tuple[str, ...] = ()

    def packet(self) -> BranchObservationData:
        return {
            "kind": "branch",
            "repository": self.repository,
            "branch": self.branch,
            "pr": self.pr.data() if self.pr else None,
            "checked_at": self.checked_at,
            "error": self.error,
            "last_success_at": self.last_success_at,
            "resolutions": self.resolutions,
        }


@dataclass(frozen=True)
class PullRequestObservation:
    ref: str
    pr: PullRequest | None
    checked_at: float
    error: str | None = None
    last_success_at: float | None = None

    def packet(self) -> PullRequestObservationData:
        return {
            "kind": "pull_request",
            "ref": self.ref,
            "pr": self.pr.data() if self.pr else None,
            "checked_at": self.checked_at,
            "error": self.error,
            "last_success_at": self.last_success_at,
        }


type RepositoryObservation = (
    CheckoutObservation | BranchObservation | PullRequestObservation
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


class CheckoutObservationData(TypedDict):
    kind: Literal["checkout"]
    worktree: str
    local: GitObservationData | None
    checked_at: float
    error: str | None
    last_success_at: float | None
    available: bool


class BranchObservationData(TypedDict):
    kind: Literal["branch"]
    repository: str
    branch: str
    pr: PullRequestData | None
    checked_at: float
    error: str | None
    last_success_at: float | None
    resolutions: tuple[str, ...]


class PullRequestObservationData(TypedDict):
    kind: Literal["pull_request"]
    ref: str
    pr: PullRequestData | None
    checked_at: float
    error: str | None
    last_success_at: float | None


type RepositoryObservationData = (
    CheckoutObservationData | BranchObservationData | PullRequestObservationData
)


def repository_observation_from_mapping(
    data: RepositoryObservationData,
) -> RepositoryObservation:
    if data["kind"] == "checkout":
        local = data["local"]
        return CheckoutObservation(
            data["worktree"],
            GitObservation(**local) if local else None,
            data["checked_at"],
            data["error"],
            data["available"],
            data.get("last_success_at"),
        )
    pr = data["pr"]
    observed_pr = pull_request_from_data(pr) if pr else None
    if data["kind"] == "branch":
        return BranchObservation(
            data["repository"],
            data["branch"],
            observed_pr,
            data["checked_at"],
            data["error"],
            data.get("last_success_at"),
            tuple(data["resolutions"]),
        )
    return PullRequestObservation(
        data["ref"],
        observed_pr,
        data["checked_at"],
        data["error"],
        data.get("last_success_at"),
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
        is_draft=pr["is_draft"],
        head=pr["head"],
        head_repository=pr["head_repository"],
        head_branch=pr["head_branch"],
        repository=pr["repository"],
        title=pr["title"],
        body=pr["body"],
        base_branch=pr["base_branch"],
        base_head=pr["base_head"],
        related_links=tuple(
            related_link_from_data(link) for link in pr["related_links"]
        ),
        related_links_error=pr["related_links_error"],
    )
