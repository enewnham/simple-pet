from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from ..domain import (
    CollectionKind,
    EvidenceKind,
    EvidenceSource,
    PullRequestState,
)
from ..identity import SessionIdentity
from ..repository_identity import RemoteRepository, RepositoryIdentity
from .models import RelatedLink, ReportPayload


class SessionObservation(BaseModel):
    session: SessionIdentity
    aliases: tuple[SessionIdentity, ...] = ()
    parent: SessionIdentity | None = None
    location: str | None = None
    reports: tuple[ReportPayload, ...] = ()
    repository: RepositoryIdentity | None = None
    branch: str | None = None
    pull_request: str | None = None


class GitObservation(BaseModel):
    type: Literal["GitObservation"] = "GitObservation"
    source: Literal[EvidenceSource.GIT] = EvidenceSource.GIT
    kind: Literal[EvidenceKind.OBSERVATION] = EvidenceKind.OBSERVATION

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
    repository: RepositoryIdentity

    @property
    def has_unpublished_changes(self) -> bool:
        return bool(
            self.staged
            or self.unstaged
            or self.untracked
            or self.conflicts
            or self.ahead
        )


class PullRequest(BaseModel):
    number: int
    url: str
    state: PullRequestState
    merged_at: str | None
    is_draft: bool | None = None
    head: str = ""
    head_repository: RepositoryIdentity | None = None
    head_branch: str = ""
    repository: RemoteRepository
    title: str = ""
    body: str = ""
    base_branch: str = ""
    base_head: str = ""
    related_links: tuple[RelatedLink, ...] = ()
    related_links_error: str | None = None

    @property
    def is_open(self) -> bool:
        return self.state == PullRequestState.OPEN


class CheckoutObservation(BaseModel):
    kind: Literal["checkout"] = "checkout"
    worktree: str
    local: GitObservation | None
    checked_at: float
    error: str | None = None
    available: bool = True
    last_success_at: float | None = None


class BranchObservation(BaseModel):
    kind: Literal["branch"] = "branch"
    repository: RepositoryIdentity
    branch: str
    pr: PullRequest | None
    checked_at: float
    error: str | None = None
    last_success_at: float | None = None
    # Retain observed PR resolutions so branch reuse cannot reattribute reports.
    resolutions: tuple[str, ...] = ()


class PullRequestObservation(BaseModel):
    kind: Literal["pull_request"] = "pull_request"
    ref: str
    pr: PullRequest | None
    checked_at: float
    error: str | None = None
    last_success_at: float | None = None


type RepositoryObservation = Annotated[
    CheckoutObservation | BranchObservation | PullRequestObservation,
    Field(discriminator="kind"),
]


class RepositoryTarget(BaseModel):
    kind: CollectionKind
    ref: str
    repository: RepositoryIdentity | None = None
    checkout: str = ""
    include_links: bool = True


repository_observation_adapter = TypeAdapter(RepositoryObservation)


from . import models as _models

for _model in (
    _models.GitHubEvidence,
    _models.EvidenceRecord,
    _models.SessionReport,
    _models.SessionRecord,
):
    _model.model_rebuild(_types_namespace=globals() | vars(_models))
