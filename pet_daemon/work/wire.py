from typing import TypedDict


class RelatedLinkData(TypedDict):
    kind: str
    ref: str
    relationship: str
    source: str
    stack: int | None
    position: int | None
    state: str | None
    is_draft: bool | None
    head: str
    head_branch: str
    title: str


class PullRequestData(TypedDict):
    number: int
    url: str
    state: str
    merged_at: str | None
    review: str
    checks: tuple[tuple[str, ...], ...]
    is_draft: bool | None
    head: str
    head_repository: str
    head_branch: str
    review_requests: tuple[str, ...]
    repository: str
    check_revision: str
    review_requested: bool | None
    title: str
    body: str
    base_branch: str
    base_head: str
    related_links: tuple[RelatedLinkData, ...]
    related_links_error: str | None
