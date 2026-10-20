from dataclasses import dataclass


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

    @property
    def pending(self):
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


@dataclass(frozen=True)
class RepositoryObservation:
    subject: str
    worktree: str
    branch: str
    sessions: tuple = ()
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

    @property
    def observed_at(self):
        return max(self.local_checked_at, self.github_checked_at)

    @property
    def pending(self):
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


def repository_observation_from_mapping(data):
    values = dict(data)
    values["sessions"] = tuple(values.get("sessions", ()))
    if values.get("local"):
        values["local"] = GitObservation(**values["local"])
    if values.get("pr"):
        pr = dict(values["pr"])
        pr["checks"] = tuple(tuple(check) for check in pr["checks"])
        pr["review_requests"] = tuple(pr.get("review_requests", ()))
        pr["related_links"] = tuple(pr.get("related_links", ()))
        values["pr"] = PullRequest(**pr)
    return RepositoryObservation(**values)
