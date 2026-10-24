from __future__ import annotations

import re
from collections.abc import Mapping
from urllib.parse import urlsplit, urlunsplit

from ..domain import ArtifactKind
from .models import ArtifactReference, ReportContext
from .observations import (
    BranchObservation,
    CheckoutObservation,
    PullRequestObservation,
    RepositoryObservation,
)
from .validation import normalize_reference


def references(prose: str) -> tuple[ArtifactReference, ...]:
    found = {}
    for matched in re.findall(r"https?://[^\s<>\"'`]+", prose):
        parsed = urlsplit(matched.rstrip(".,;:!?)]}"))
        path = parsed.path.rstrip("/")
        if re.fullmatch(r"/[^/]+/[^/]+/pull/\d+", path):
            kind = ArtifactKind.PULL_REQUEST
        elif (
            parsed.hostname == "linear.app"
            and re.search(r"/issue/[^/]+", path)
            or parsed.hostname == "github.com"
            and re.fullmatch(r"/[^/]+/[^/]+/issues/\d+", path)
            or re.fullmatch(r"/browse/[A-Z][A-Z0-9]+-\d+", path)
        ):
            kind = ArtifactKind.TICKET
        else:
            continue
        ref = urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, "", ""))
        found[kind, ref] = normalize_reference(ArtifactReference(kind=kind, ref=ref))
    return tuple(found.values())


def report_context(
    location: str,
    checkout: CheckoutObservation | None,
    branch: BranchObservation | None,
    prose: str = "",
) -> ReportContext:
    local = checkout.local if checkout else None
    return ReportContext(
        checkout=checkout.worktree if checkout else location,
        repository=local.repository if local else None,
        branch=local.branch if local else "",
        pull_request=branch.pr.url if branch and branch.pr else "",
        references=references(prose),
        branch_resolution=len(branch.resolutions)
        if branch and branch.pr is None
        else 0,
    )


TERMINAL_EVENTS = {
    "turn_done",
    "turn_error",
    "subagent_done",
    "interrupt",
    "session_end",
}


def enclosing_checkout(
    checkouts: Mapping[str, CheckoutObservation], location: str
) -> CheckoutObservation | None:
    matches = [
        checkout
        for path, checkout in checkouts.items()
        if location == path or location.startswith(path.rstrip("/") + "/")
    ]
    return (
        max(matches, key=lambda checkout: len(checkout.worktree)) if matches else None
    )


def reconcile_observation(
    record: RepositoryObservation | None, observation: RepositoryObservation
) -> RepositoryObservation | None:
    if record and observation.checked_at < record.checked_at:
        return None
    failed = (
        bool(observation.error)
        or isinstance(observation, CheckoutObservation)
        and not observation.available
    )
    last_success_at = record.last_success_at if record else None
    if not failed:
        last_success_at = observation.checked_at
    observation = observation.model_copy()
    observation.last_success_at = last_success_at
    if record and failed:
        match observation, record:
            case CheckoutObservation(), CheckoutObservation():
                observation.local = record.local
            case BranchObservation(), BranchObservation():
                observation.pr = record.pr
            case PullRequestObservation(), PullRequestObservation():
                observation.pr = record.pr
    if (
        isinstance(observation, (BranchObservation, PullRequestObservation))
        and isinstance(record, (BranchObservation, PullRequestObservation))
        and observation.pr
        and record.pr
        and observation.pr.url == record.pr.url
        and observation.pr.related_links_error
    ):
        observation.pr = observation.pr.model_copy()
        observation.pr.related_links = record.pr.related_links
    if isinstance(observation, BranchObservation):
        prior = record if isinstance(record, BranchObservation) else None
        resolutions = prior.resolutions if prior else ()
        if (
            not failed
            and observation.pr
            and (
                prior is None or prior.pr is None or prior.pr.url != observation.pr.url
            )
        ):
            resolutions = (*resolutions, observation.pr.url)
        observation.resolutions = resolutions
    return observation if record != observation else None
