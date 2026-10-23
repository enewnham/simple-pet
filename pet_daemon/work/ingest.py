from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import Protocol

from ..domain import ArtifactKind
from .association import TERMINAL_EVENTS, decide_observation, report_context
from .models import ArtifactReference, ExecutionEvidence, ReportContext, SessionReport
from .observations import (
    BranchObservation,
    CheckoutObservation,
    PullRequestObservation,
    RepositoryObservation,
    SessionObservation,
)
from .validation import normalize_reference, repository_identity


class WorkRepository(Protocol):
    def transaction(self) -> AbstractContextManager[None]: ...
    def save_session(self, observation: SessionObservation, now: float) -> None: ...
    def append_report(self, report: SessionReport) -> bool: ...
    def session_location(self, session: str) -> str: ...
    def get_checkout(
        self, path: str, *, enclosing: bool = False
    ) -> CheckoutObservation | None: ...
    def get_branch(self, repository: str, branch: str) -> BranchObservation | None: ...
    def get_pull_request(self, ref: str) -> PullRequestObservation | None: ...
    def put_observation(self, observation: RepositoryObservation) -> None: ...
    def clear(self) -> None: ...


class WorkIngestion:
    def __init__(
        self, repository: WorkRepository, clock: Callable[[], float] = time.time
    ) -> None:
        self.repository = repository
        self.clock = clock

    def _context(
        self,
        location: str,
        prose: str = "",
        *,
        repository: str | None = None,
        branch: str | None = None,
        pull_request: str | None = None,
    ) -> ReportContext:
        if pull_request:
            pull_request = normalize_reference(
                ArtifactReference(ArtifactKind.PULL_REQUEST, pull_request)
            ).ref
        checkout = (
            self.repository.get_checkout(location, enclosing=True) if location else None
        )
        local = checkout.local if checkout else None
        repo = repository_identity(repository or (local.repository if local else ""))
        branch_name = branch or (local.branch if local else "")
        observed = (
            self.repository.get_branch(repo, branch_name)
            if repo and branch_name
            else None
        )
        context = report_context(location, checkout, observed, prose)
        explicit = next(
            (
                item.ref
                for item in context.references
                if item.kind == ArtifactKind.PULL_REQUEST
            ),
            "",
        )
        resolved_pr = pull_request or explicit or context.pull_request
        return replace(
            context,
            repository=repo,
            branch=branch_name,
            pull_request=resolved_pr,
            branch_resolution=len(observed.resolutions)
            if observed and not resolved_pr
            else 0,
        )

    def record_session(self, observation: SessionObservation) -> None:
        now = self.clock()
        with self.repository.transaction():
            self.repository.save_session(observation, now)
            location = (
                observation.location
                if observation.location is not None
                else self.repository.session_location(observation.session)
            )
            contexts = tuple(
                self._context(
                    location,
                    (payload.progress or "")
                    if isinstance(payload, ExecutionEvidence)
                    else payload.text,
                    repository=observation.repository,
                    branch=observation.branch,
                    pull_request=observation.pull_request,
                )
                for payload in observation.reports
            )
            for payload, context in zip(observation.reports, contexts, strict=True):
                appended = self.repository.append_report(
                    SessionReport(payload, context, now)
                )
                if (
                    not appended
                    and isinstance(payload, ExecutionEvidence)
                    and payload.event in TERMINAL_EVENTS - {"session_end"}
                ):
                    break

    def observe(self, collected: RepositoryObservation) -> None:
        with self.repository.transaction():
            match collected:
                case CheckoutObservation():
                    record = self.repository.get_checkout(collected.worktree)
                case BranchObservation():
                    collected = replace(
                        collected, repository=repository_identity(collected.repository)
                    )
                    record = self.repository.get_branch(
                        collected.repository, collected.branch
                    )
                case PullRequestObservation():
                    record = self.repository.get_pull_request(collected.ref)
            observation = decide_observation(record, collected)
            if observation is None:
                return
            self.repository.put_observation(observation)
            if isinstance(collected, BranchObservation) and collected.pr:
                self.observe(
                    PullRequestObservation(
                        collected.pr.url,
                        collected.pr,
                        collected.checked_at,
                        collected.error,
                    )
                )

    def clear(self) -> None:
        with self.repository.transaction():
            self.repository.clear()
